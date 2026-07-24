from pathlib import Path
import argparse
from contextlib import nullcontext
from collections import defaultdict
import random
import copy

import torch
from torch.utils.data import DataLoader, Subset
from torchvision.ops import batched_nms

from dataset import VOCDatasetYOLO, detection_collate_fn, VOC_CLASSES
from model import (
    TinyYOLOAnchorFree,
    count_parameters,
    load_inference_state_dict,
)
from predict import nms


EXPECTED_EXPERIMENT_NAME = "voc8_v08_768_mobilenetv3_distilled"


def inference_autocast(device: torch.device, precision: str):
    if precision == "fp16" and device.type == "cuda":
        return torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
        )

    return nullcontext()


def move_images_to_device(
    images: torch.Tensor,
    device: torch.device,
    memory_format: str,
) -> torch.Tensor:
    images = images.to(device, non_blocking=True)

    if memory_format == "channels-last":
        images = images.contiguous(memory_format=torch.channels_last)

    return images


def validate_checkpoint_metadata(
    checkpoint: dict,
    checkpoint_path: Path,
    expected_experiment_name: str,
    expected_inference_parameter_count: int,
) -> None:
    checkpoint_experiment = checkpoint.get("experiment_name")

    if checkpoint_experiment != expected_experiment_name:
        raise ValueError(
            "Checkpoint does not belong to the current experiment.\n"
            f"Checkpoint: {checkpoint_path}\n"
            f"Expected experiment: {expected_experiment_name!r}\n"
            f"Found experiment:    {checkpoint_experiment!r}\n"
            "Run python src\\train.py to create a current checkpoint."
        )

    checkpoint_parameter_count = checkpoint.get("inference_parameter_count")

    if checkpoint_parameter_count is not None:
        checkpoint_parameter_count = int(checkpoint_parameter_count)

        if checkpoint_parameter_count != expected_inference_parameter_count:
            raise ValueError(
                "Checkpoint parameter count does not match the current model.\n"
                f"Checkpoint: {checkpoint_path}\n"
                f"Expected inference params: {expected_inference_parameter_count}\n"
                f"Found inference params:    {checkpoint_parameter_count}\n"
                "Run python src\\train.py to create a current checkpoint."
            )


SIZE_BUCKETS = {
    "small": (0.0, 32.0 * 32.0),
    "medium": (32.0 * 32.0, 96.0 * 96.0),
    "large": (96.0 * 96.0, float("inf")),
}


def get_size_name_from_area(area: float):
    for size_name, (min_area, max_area) in SIZE_BUCKETS.items():
        if area >= min_area and area < max_area:
            return size_name

    return "large"


def targets_to_xyxy(targets: torch.Tensor, image_size: int):
    if targets.numel() == 0:
        return torch.zeros((0, 5), dtype=torch.float32, device=targets.device)

    class_ids = targets[:, 0]

    x_center = targets[:, 1] * image_size
    y_center = targets[:, 2] * image_size
    width = targets[:, 3] * image_size
    height = targets[:, 4] * image_size

    x1 = x_center - width / 2
    y1 = y_center - height / 2
    x2 = x_center + width / 2
    y2 = y_center + height / 2

    return torch.stack([class_ids, x1, y1, x2, y2], dim=1)


def box_area_xyxy(boxes: torch.Tensor):
    if boxes.numel() == 0:
        return torch.zeros((0,), dtype=torch.float32, device=boxes.device)

    widths = (boxes[:, 2] - boxes[:, 0]).clamp(min=0)
    heights = (boxes[:, 3] - boxes[:, 1]).clamp(min=0)

    return widths * heights


def box_iou_matrix(boxes1: torch.Tensor, boxes2: torch.Tensor, eps: float = 1e-7):
    if boxes1.numel() == 0 or boxes2.numel() == 0:
        return torch.zeros((boxes1.shape[0], boxes2.shape[0]), device=boxes1.device)

    area1 = box_area_xyxy(boxes1)
    area2 = box_area_xyxy(boxes2)

    inter_x1 = torch.maximum(boxes1[:, None, 0], boxes2[None, :, 0])
    inter_y1 = torch.maximum(boxes1[:, None, 1], boxes2[None, :, 1])
    inter_x2 = torch.minimum(boxes1[:, None, 2], boxes2[None, :, 2])
    inter_y2 = torch.minimum(boxes1[:, None, 3], boxes2[None, :, 3])

    inter_w = (inter_x2 - inter_x1).clamp(min=0)
    inter_h = (inter_y2 - inter_y1).clamp(min=0)
    inter_area = inter_w * inter_h

    union = area1[:, None] + area2[None, :] - inter_area

    return inter_area / (union + eps)


def box_iou_one_to_many(box: torch.Tensor, boxes: torch.Tensor, eps: float = 1e-7):
    if boxes.numel() == 0:
        return torch.zeros((0,), dtype=torch.float32, device=box.device)

    x1 = torch.maximum(box[0], boxes[:, 0])
    y1 = torch.maximum(box[1], boxes[:, 1])
    x2 = torch.minimum(box[2], boxes[:, 2])
    y2 = torch.minimum(box[3], boxes[:, 3])

    inter_w = (x2 - x1).clamp(min=0)
    inter_h = (y2 - y1).clamp(min=0)
    inter_area = inter_w * inter_h

    box_area = (box[2] - box[0]).clamp(min=0) * (box[3] - box[1]).clamp(min=0)

    boxes_area = (boxes[:, 2] - boxes[:, 0]).clamp(min=0) * (
        boxes[:, 3] - boxes[:, 1]
    ).clamp(min=0)

    union = box_area + boxes_area - inter_area

    return inter_area / (union + eps)


def suppress_cross_class_duplicate_boxes(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    class_ids: torch.Tensor,
    duplicate_iou_threshold: float = 0.90,
    size_similarity_threshold: float = 0.95,
    center_distance_threshold: float = 0.05,
):
    """
    Removes duplicate-looking boxes even if they have different class labels.

    A lower-confidence box is removed only if it is almost the same box as
    a higher-confidence box.

    Conditions:
    1. IoU must be at least duplicate_iou_threshold.
    2. Width similarity must be at least size_similarity_threshold.
    3. Height similarity must be at least size_similarity_threshold.
    4. Center distance must be very small compared with box size.

    This is stricter than full class-agnostic NMS.
    """

    if boxes.numel() == 0:
        return boxes, scores, class_ids

    order = scores.argsort(descending=True)

    boxes = boxes[order]
    scores = scores[order]
    class_ids = class_ids[order]

    keep_indices = []

    widths = (boxes[:, 2] - boxes[:, 0]).clamp(min=1e-6)
    heights = (boxes[:, 3] - boxes[:, 1]).clamp(min=1e-6)

    centers_x = (boxes[:, 0] + boxes[:, 2]) / 2.0
    centers_y = (boxes[:, 1] + boxes[:, 3]) / 2.0

    diagonals = torch.sqrt(widths * widths + heights * heights).clamp(min=1e-6)

    for current_idx in range(boxes.shape[0]):
        current_box = boxes[current_idx]
        current_class = class_ids[current_idx]

        should_keep = True

        for kept_idx in keep_indices:
            kept_box = boxes[kept_idx]
            kept_class = class_ids[kept_idx]

            # Same-class duplicates were already handled by normal class-wise NMS.
            # This only targets different-class duplicate boxes.
            if current_class == kept_class:
                continue

            iou = box_iou_one_to_many(
                box=current_box,
                boxes=kept_box.unsqueeze(0),
            )[0]

            if iou < duplicate_iou_threshold:
                continue

            current_width = widths[current_idx]
            current_height = heights[current_idx]
            kept_width = widths[kept_idx]
            kept_height = heights[kept_idx]

            width_similarity = torch.minimum(current_width, kept_width) / torch.maximum(
                current_width,
                kept_width,
            )

            height_similarity = torch.minimum(current_height, kept_height) / torch.maximum(
                current_height,
                kept_height,
            )

            if width_similarity < size_similarity_threshold:
                continue

            if height_similarity < size_similarity_threshold:
                continue

            center_dx = centers_x[current_idx] - centers_x[kept_idx]
            center_dy = centers_y[current_idx] - centers_y[kept_idx]
            center_distance = torch.sqrt(center_dx * center_dx + center_dy * center_dy)

            smaller_diagonal = torch.minimum(diagonals[current_idx], diagonals[kept_idx])
            normalized_center_distance = center_distance / smaller_diagonal

            if normalized_center_distance > center_distance_threshold:
                continue

            # All checks passed:
            # Same area, same size, very close center, different class.
            # Since boxes are sorted by score, the kept box has higher confidence.
            should_keep = False
            break

        if should_keep:
            keep_indices.append(current_idx)

    keep_indices = torch.tensor(
        keep_indices,
        dtype=torch.long,
        device=boxes.device,
    )

    return boxes[keep_indices], scores[keep_indices], class_ids[keep_indices]


def collect_predictions_per_image(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    conf_threshold: float,
    nms_iou_threshold: float,
    max_detections_per_image: int,
    use_cross_class_duplicate_suppression: bool = True,
    duplicate_iou_threshold: float = 0.90,
    size_similarity_threshold: float = 0.95,
    center_distance_threshold: float = 0.05,
    nms_backend: str = "python",
):
    if nms_backend not in {"python", "torchvision"}:
        raise ValueError(
            f"Unsupported NMS backend: {nms_backend!r}. "
            "Use 'python' or 'torchvision'."
        )

    if nms_backend == "torchvision":
        candidate_mask = scores >= conf_threshold
        point_indices, class_ids = candidate_mask.nonzero(as_tuple=True)

        if point_indices.numel() == 0:
            return (
                torch.zeros((0, 4), dtype=torch.float32, device=boxes.device),
                torch.zeros((0,), dtype=torch.float32, device=boxes.device),
                torch.zeros((0,), dtype=torch.long, device=boxes.device),
            )

        final_boxes = boxes[point_indices].float()
        final_scores = scores[point_indices, class_ids].float()
        final_classes = class_ids.long()

        keep_indices = batched_nms(
            boxes=final_boxes,
            scores=final_scores,
            idxs=final_classes,
            iou_threshold=nms_iou_threshold,
        )

        final_boxes = final_boxes[keep_indices]
        final_scores = final_scores[keep_indices]
        final_classes = final_classes[keep_indices]
    else:
        all_boxes = []
        all_scores = []
        all_classes = []

        num_classes = scores.shape[1]

        for class_id in range(num_classes):
            class_scores = scores[:, class_id]
            keep = class_scores >= conf_threshold

            if keep.sum() == 0:
                continue

            class_boxes = boxes[keep].float()
            class_scores = class_scores[keep].float()

            keep_indices = nms(
                boxes=class_boxes,
                scores=class_scores,
                iou_threshold=nms_iou_threshold,
            )

            class_boxes = class_boxes[keep_indices]
            class_scores = class_scores[keep_indices]

            class_ids = torch.full(
                (class_boxes.shape[0],),
                class_id,
                dtype=torch.long,
                device=boxes.device,
            )

            all_boxes.append(class_boxes)
            all_scores.append(class_scores)
            all_classes.append(class_ids)

        if not all_boxes:
            return (
                torch.zeros((0, 4), dtype=torch.float32, device=boxes.device),
                torch.zeros((0,), dtype=torch.float32, device=boxes.device),
                torch.zeros((0,), dtype=torch.long, device=boxes.device),
            )

        final_boxes = torch.cat(all_boxes, dim=0)
        final_scores = torch.cat(all_scores, dim=0)
        final_classes = torch.cat(all_classes, dim=0)

    if use_cross_class_duplicate_suppression:
        final_boxes, final_scores, final_classes = suppress_cross_class_duplicate_boxes(
            boxes=final_boxes,
            scores=final_scores,
            class_ids=final_classes,
            duplicate_iou_threshold=duplicate_iou_threshold,
            size_similarity_threshold=size_similarity_threshold,
            center_distance_threshold=center_distance_threshold,
        )

    order = final_scores.argsort(descending=True)

    if order.numel() > max_detections_per_image:
        order = order[:max_detections_per_image]

    return final_boxes[order], final_scores[order], final_classes[order]


def compute_ap(recall: torch.Tensor, precision: torch.Tensor):
    mrec = torch.cat(
        [
            torch.tensor([0.0], device=recall.device),
            recall,
            torch.tensor([1.0], device=recall.device),
        ]
    )

    mpre = torch.cat(
        [
            torch.tensor([0.0], device=precision.device),
            precision,
            torch.tensor([0.0], device=precision.device),
        ]
    )

    for i in range(mpre.numel() - 2, -1, -1):
        mpre[i] = torch.maximum(mpre[i], mpre[i + 1])

    changing_points = torch.where(mrec[1:] != mrec[:-1])[0]

    ap = torch.sum(
        (mrec[changing_points + 1] - mrec[changing_points])
        * mpre[changing_points + 1]
    )

    return ap.item()


def evaluate_map(
    predictions_by_class,
    ground_truths_by_class,
    num_classes: int,
    iou_threshold: float = 0.5,
):
    per_class_results = []
    aps = []

    total_tp_all = 0
    total_fp_all = 0
    total_gt_all = 0

    for class_id in range(num_classes):
        preds = predictions_by_class[class_id]
        gts = ground_truths_by_class[class_id]

        num_gt = sum(len(v) for v in gts.values())
        total_gt_all += num_gt

        if num_gt == 0:
            continue

        preds = sorted(preds, key=lambda x: x["score"], reverse=True)

        matched = {
            image_id: torch.zeros(len(gt_boxes), dtype=torch.bool)
            for image_id, gt_boxes in gts.items()
        }

        tp = torch.zeros(len(preds))
        fp = torch.zeros(len(preds))

        for pred_idx, pred in enumerate(preds):
            image_id = pred["image_id"]
            pred_box = pred["box"].unsqueeze(0)

            if image_id not in gts:
                fp[pred_idx] = 1
                continue

            gt_boxes = gts[image_id]

            if len(gt_boxes) == 0:
                fp[pred_idx] = 1
                continue

            ious = box_iou_matrix(pred_box, gt_boxes).squeeze(0)

            best_iou, best_gt_idx = ious.max(dim=0)

            if best_iou >= iou_threshold and not matched[image_id][best_gt_idx]:
                tp[pred_idx] = 1
                matched[image_id][best_gt_idx] = True
            else:
                fp[pred_idx] = 1

        total_tp = tp.sum().item()
        total_fp = fp.sum().item()

        total_tp_all += total_tp
        total_fp_all += total_fp

        tp_cumsum = torch.cumsum(tp, dim=0)
        fp_cumsum = torch.cumsum(fp, dim=0)

        recall = tp_cumsum / max(num_gt, 1)
        precision = tp_cumsum / torch.clamp(tp_cumsum + fp_cumsum, min=1e-7)

        ap = compute_ap(recall, precision)
        aps.append(ap)

        final_precision = precision[-1].item() if precision.numel() > 0 else 0.0
        final_recall = recall[-1].item() if recall.numel() > 0 else 0.0

        per_class_results.append(
            {
                "class_id": class_id,
                "class_name": VOC_CLASSES[class_id],
                "num_gt": num_gt,
                "num_predictions": len(preds),
                "precision": final_precision,
                "recall": final_recall,
                "ap": ap,
            }
        )

    mean_ap = sum(aps) / len(aps) if aps else 0.0

    overall_precision = total_tp_all / max(total_tp_all + total_fp_all, 1)
    overall_recall = total_tp_all / max(total_gt_all, 1)

    return mean_ap, overall_precision, overall_recall, per_class_results



def evaluate_map_range(
    predictions_by_class,
    ground_truths_by_class,
    num_classes: int,
    iou_thresholds: list[float],
):
    threshold_results = {}

    ap_sums_by_class = defaultdict(float)
    ap_counts_by_class = defaultdict(int)

    map_values = []
    results_at_50 = None

    for threshold in iou_thresholds:
        mean_ap, precision, recall, per_class_results = evaluate_map(
            predictions_by_class=predictions_by_class,
            ground_truths_by_class=ground_truths_by_class,
            num_classes=num_classes,
            iou_threshold=threshold,
        )

        threshold_results[threshold] = {
            "mean_ap": mean_ap,
            "precision": precision,
            "recall": recall,
            "per_class_results": per_class_results,
        }

        map_values.append(mean_ap)

        if abs(threshold - 0.50) < 1e-6:
            results_at_50 = threshold_results[threshold]

        for result in per_class_results:
            class_id = result["class_id"]
            ap_sums_by_class[class_id] += result["ap"]
            ap_counts_by_class[class_id] += 1

    if results_at_50 is None:
        results_at_50 = threshold_results[iou_thresholds[0]]

    per_class_ap_50_95 = {}

    for class_id in range(num_classes):
        if ap_counts_by_class[class_id] == 0:
            per_class_ap_50_95[class_id] = 0.0
        else:
            per_class_ap_50_95[class_id] = (
                ap_sums_by_class[class_id] / ap_counts_by_class[class_id]
            )

    map_50_95 = sum(map_values) / max(len(map_values), 1)

    return {
        "map50": results_at_50["mean_ap"],
        "map50_95": map_50_95,
        "precision50": results_at_50["precision"],
        "recall50": results_at_50["recall"],
        "per_class_results_50": results_at_50["per_class_results"],
        "per_class_ap_50_95": per_class_ap_50_95,
        "threshold_results": threshold_results,
    }


def build_size_ground_truths(gt_xyxy: torch.Tensor, num_classes: int):
    size_ground_truths = {
        size_name: {}
        for size_name in SIZE_BUCKETS.keys()
    }

    for size_name in SIZE_BUCKETS.keys():
        for class_id in range(num_classes):
            size_ground_truths[size_name][class_id] = torch.zeros(
                (0, 4),
                dtype=torch.float32,
                device=gt_xyxy.device,
            )

    if gt_xyxy.numel() == 0:
        return size_ground_truths

    for class_id in range(num_classes):
        class_mask = gt_xyxy[:, 0].long() == class_id
        class_gt_boxes = gt_xyxy[class_mask][:, 1:5]

        if class_gt_boxes.numel() == 0:
            continue

        areas = box_area_xyxy(class_gt_boxes)

        for size_name, (min_area, max_area) in SIZE_BUCKETS.items():
            size_mask = (areas >= min_area) & (areas < max_area)
            size_ground_truths[size_name][class_id] = class_gt_boxes[size_mask]

    return size_ground_truths


def filter_predictions_by_predicted_size(
    predictions_by_class,
    num_classes: int,
    size_name: str,
):
    min_area, max_area = SIZE_BUCKETS[size_name]

    filtered_predictions = defaultdict(list)

    for class_id in range(num_classes):
        for pred in predictions_by_class[class_id]:
            box = pred["box"].unsqueeze(0)
            area = box_area_xyxy(box).item()

            if area >= min_area and area < max_area:
                filtered_predictions[class_id].append(copy.deepcopy(pred))

    return filtered_predictions


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=str, default="data/processed/voc2007_2012_custom_voc8")
    parser.add_argument("--checkpoint", type=str, default="runs/checkpoints_voc8_v08_768_mobilenetv3_distilled/best.pt")
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--image-size", type=int, default=768)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-images", type=int, default=0)
    parser.add_argument("--conf", type=float, default=0.10)
    parser.add_argument("--nms-iou", type=float, default=0.50)
    parser.add_argument("--max-det", type=int, default=100)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument(
        "--precision",
        choices=["fp32", "fp16"],
        default="fp32",
        help="Use fp32 for published metrics. Use fp16 to test optimized inference accuracy.",
    )
    parser.add_argument(
        "--memory-format",
        choices=["contiguous", "channels-last"],
        default="contiguous",
    )
    parser.add_argument(
        "--nms-backend",
        choices=["python", "torchvision"],
        default="python",
        help="python preserves the original evaluation path; torchvision is faster.",
    )
    parser.add_argument("--compile", dest="compile_model", action="store_true")
    parser.add_argument(
        "--compile-mode",
        choices=["default", "reduce-overhead", "max-autotune"],
        default="reduce-overhead",
    )
    parser.add_argument(
        "--disable-cross-class-duplicate-suppression",
        action="store_true",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]

    dataset_root = Path(args.dataset_root)
    checkpoint_path = Path(args.checkpoint)
    if not dataset_root.is_absolute():
        dataset_root = project_root / dataset_root
    if not checkpoint_path.is_absolute():
        checkpoint_path = project_root / checkpoint_path

    # "val"  = VOC2007 test, used as validation during training
    # "test" = VOC2012 val, used as final test split
    split = args.split

    image_size = args.image_size
    num_classes = len(VOC_CLASSES)
    batch_size = args.batch_size

    max_eval_images = args.max_images if args.max_images > 0 else None
    use_random_subset = True
    random_seed = args.random_seed

    conf_threshold = args.conf

    nms_iou_threshold = args.nms_iou
    max_detections_per_image = args.max_det
    iou_match_threshold = 0.50

    use_cross_class_duplicate_suppression = (
        not args.disable_cross_class_duplicate_suppression
    )
    duplicate_iou_threshold = 0.90
    size_similarity_threshold = 0.95
    center_distance_threshold = 0.05

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.precision == "fp16" and device.type != "cuda":
        raise RuntimeError("FP16 evaluation requires a CUDA device.")

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    print(f"Device: {device}")
    print(f"Dataset root: {dataset_root}")
    print(f"Dataset split: {split}")
    print('Split meaning: "val" = VOC2007 test, "test" = VOC2012 val')
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Expected experiment: {EXPECTED_EXPERIMENT_NAME}")
    print(f"Classes: {VOC_CLASSES}")
    print()
    print("Evaluation filtering:")
    print(f"Confidence threshold: {conf_threshold}")
    print(f"Class-wise NMS IoU threshold: {nms_iou_threshold}")
    print(f"Max detections/image: {max_detections_per_image}")
    print(f"Cross-class duplicate suppression: {use_cross_class_duplicate_suppression}")
    print(f"Duplicate IoU threshold: {duplicate_iou_threshold}")
    print(f"Size similarity threshold: {size_similarity_threshold}")
    print(f"Center distance threshold: {center_distance_threshold}")
    print(f"Precision: {args.precision}")
    print(f"Memory format: {args.memory_format}")
    print(f"NMS backend: {args.nms_backend}")
    print(f"torch.compile: {args.compile_model}")
    if args.compile_model:
        print(f"Compile mode: {args.compile_mode}")
    print()

    full_dataset = VOCDatasetYOLO(
        root_dir=dataset_root,
        split=split,
        image_size=image_size,
        augment=False,
        mosaic_prob=0.0,
    )

    if max_eval_images is not None and max_eval_images < len(full_dataset):
        if use_random_subset:
            if random_seed is not None:
                random.seed(random_seed)

            subset_indices = random.sample(
                population=range(len(full_dataset)),
                k=max_eval_images,
            )
        else:
            subset_indices = list(range(max_eval_images))

        dataset = Subset(full_dataset, subset_indices)

        print(f"Full split images: {len(full_dataset)}")
        print(f"Using subset:      {len(dataset)} images")
        print(f"Random subset:     {use_random_subset}")
        print(f"Random seed:       {random_seed}")
        print()
    else:
        dataset = full_dataset

        print(f"Using full dataset: {len(dataset)} images")
        print()

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=detection_collate_fn,
        num_workers=0,
        pin_memory=True if device.type == "cuda" else False,
    )

    model = TinyYOLOAnchorFree(
        num_classes=num_classes,
        image_size=image_size,
        reg_max=16,
        pretrained_backbone=False,
        use_auxiliary_heads=False,
    ).to(device)

    model_parameter_count = count_parameters(model)

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}. "
            "Train first or check the checkpoint path."
        )

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    validate_checkpoint_metadata(
        checkpoint=checkpoint,
        checkpoint_path=checkpoint_path,
        expected_experiment_name=EXPECTED_EXPERIMENT_NAME,
        expected_inference_parameter_count=model_parameter_count,
    )
    load_inference_state_dict(model, checkpoint["model_state_dict"])
    model.eval()

    if args.memory_format == "channels-last":
        model = model.to(memory_format=torch.channels_last)

    if args.compile_model:
        if not hasattr(torch, "compile"):
            raise RuntimeError("This PyTorch installation does not support torch.compile.")

        model = torch.compile(
            model,
            mode=args.compile_mode,
            fullgraph=False,
        )

    print(f"Loaded epoch: {checkpoint['epoch']}")
    print(f"Checkpoint val loss: {checkpoint['val_loss']:.4f}")
    print()

    predictions_by_class = defaultdict(list)
    ground_truths_by_class = defaultdict(dict)

    ground_truths_by_size = {
        size_name: defaultdict(dict)
        for size_name in SIZE_BUCKETS.keys()
    }

    image_counter = 0

    for batch_idx, (images, targets, image_paths) in enumerate(loader, start=1):
        images = move_images_to_device(
            images=images,
            device=device,
            memory_format=args.memory_format,
        )

        with inference_autocast(device=device, precision=args.precision):
            decoded = model(images, decode=True)

        batch_boxes = decoded["boxes"].float()
        batch_scores = decoded["scores"].float()

        batch_size_actual = images.shape[0]

        for i in range(batch_size_actual):
            image_id = image_counter
            image_counter += 1

            gt = targets[i].to(device)
            gt_xyxy = targets_to_xyxy(gt, image_size=image_size)

            for class_id in range(num_classes):
                class_mask = gt_xyxy[:, 0].long() == class_id
                class_gt_boxes = gt_xyxy[class_mask][:, 1:5].detach().cpu()

                ground_truths_by_class[class_id][image_id] = class_gt_boxes

            size_gt_dict = build_size_ground_truths(
                gt_xyxy=gt_xyxy,
                num_classes=num_classes,
            )

            for size_name in SIZE_BUCKETS.keys():
                for class_id in range(num_classes):
                    ground_truths_by_size[size_name][class_id][image_id] = (
                        size_gt_dict[size_name][class_id].detach().cpu()
                    )

            boxes = batch_boxes[i]
            scores = batch_scores[i]

            final_boxes, final_scores, final_classes = collect_predictions_per_image(
                boxes=boxes,
                scores=scores,
                conf_threshold=conf_threshold,
                nms_iou_threshold=nms_iou_threshold,
                max_detections_per_image=max_detections_per_image,
                use_cross_class_duplicate_suppression=use_cross_class_duplicate_suppression,
                duplicate_iou_threshold=duplicate_iou_threshold,
                size_similarity_threshold=size_similarity_threshold,
                center_distance_threshold=center_distance_threshold,
                nms_backend=args.nms_backend,
            )

            final_boxes = final_boxes.detach().cpu()
            final_scores = final_scores.detach().cpu()
            final_classes = final_classes.detach().cpu()

            final_areas = box_area_xyxy(final_boxes)

            for box, score, class_id, area in zip(
                final_boxes,
                final_scores,
                final_classes,
                final_areas,
            ):
                class_id_int = int(class_id.item())
                area_float = float(area.item())

                predictions_by_class[class_id_int].append(
                    {
                        "image_id": image_id,
                        "score": float(score.item()),
                        "box": box,
                        "area": area_float,
                        "size_name": get_size_name_from_area(area_float),
                    }
                )

        if batch_idx % 10 == 0 or batch_idx == 1:
            print(f"Evaluated batch {batch_idx}/{len(loader)}")

    iou_thresholds = [
        round(0.50 + 0.05 * idx, 2)
        for idx in range(10)
    ]

    eval_results = evaluate_map_range(
        predictions_by_class=predictions_by_class,
        ground_truths_by_class=ground_truths_by_class,
        num_classes=num_classes,
        iou_thresholds=iou_thresholds,
    )

    print()
    print("=" * 80)
    print(f"Fast evaluation results on {split}")
    print("=" * 80)
    print(f"Images used: {len(dataset)} / {len(full_dataset)}")
    print(f"mAP50:     {eval_results['map50']:.4f}")
    print(f"mAP50-95:  {eval_results['map50_95']:.4f}")
    print(f"Precision: {eval_results['precision50']:.4f}")
    print(f"Recall:    {eval_results['recall50']:.4f}")
    print()

    print("Per-class AP:")
    for result in eval_results["per_class_results_50"]:
        class_id = result["class_id"]
        print(
            f"{result['class_name']:12s} | "
            f"AP50 {result['ap']:.4f} | "
            f"AP50-95 {eval_results['per_class_ap_50_95'][class_id]:.4f} | "
            f"P {result['precision']:.4f} | "
            f"R {result['recall']:.4f} | "
            f"GT {result['num_gt']:4d} | "
            f"Pred {result['num_predictions']:5d}"
        )

    print()
    print("=" * 80)
    print("Object-size AP")
    print("=" * 80)
    print(f"Size buckets use COCO-style pixel areas after {image_size} letterbox:")
    print("small:  area < 32^2")
    print("medium: 32^2 <= area < 96^2")
    print("large:  area >= 96^2")
    print()
    print("Fairness fix:")
    print("small AP uses small GT boxes and small predicted boxes")
    print("medium AP uses medium GT boxes and medium predicted boxes")
    print("large AP uses large GT boxes and large predicted boxes")
    print()

    for size_name in SIZE_BUCKETS.keys():
        size_predictions = filter_predictions_by_predicted_size(
            predictions_by_class=predictions_by_class,
            num_classes=num_classes,
            size_name=size_name,
        )

        size_eval_results = evaluate_map_range(
            predictions_by_class=size_predictions,
            ground_truths_by_class=ground_truths_by_size[size_name],
            num_classes=num_classes,
            iou_thresholds=iou_thresholds,
        )

        size_gt_total = sum(
            result["num_gt"]
            for result in size_eval_results["per_class_results_50"]
        )

        size_pred_total = sum(
            result["num_predictions"]
            for result in size_eval_results["per_class_results_50"]
        )

        print(
            f"{size_name:7s} | "
            f"mAP50 {size_eval_results['map50']:.4f} | "
            f"mAP50-95 {size_eval_results['map50_95']:.4f} | "
            f"P {size_eval_results['precision50']:.4f} | "
            f"R {size_eval_results['recall50']:.4f} | "
            f"GT {size_gt_total:5d} | "
            f"Pred {size_pred_total:5d}"
        )

    print()
    print("IoU-threshold mAP curve:")
    for threshold in iou_thresholds:
        threshold_result = eval_results["threshold_results"][threshold]
        print(
            f"  IoU {threshold:.2f}: "
            f"mAP {threshold_result['mean_ap']:.4f}"
        )

    print()
    print("Done.")


if __name__ == "__main__":
    main()
