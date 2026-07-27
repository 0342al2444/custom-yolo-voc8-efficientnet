from pathlib import Path
import argparse
from contextlib import nullcontext
import random

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from torchvision.ops import batched_nms

from dataset import VOC_CLASSES
from model import (
    TinyYOLOAnchorFree,
    count_parameters,
    load_inference_state_dict,
)


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


def find_image_paths(dataset_root: Path, split: str):
    possible_dirs = [
        dataset_root / split / "images",
        dataset_root / "images" / split,
        dataset_root / split,
    ]

    image_extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp"]

    for image_dir in possible_dirs:
        if image_dir.exists():
            image_paths = []

            for extension in image_extensions:
                image_paths.extend(image_dir.glob(extension))

            image_paths = sorted(image_paths)

            if image_paths:
                return image_dir, image_paths

    image_paths = []

    for extension in image_extensions:
        for path in dataset_root.rglob(extension):
            path_parts = [part.lower() for part in path.parts]

            if split.lower() in path_parts:
                image_paths.append(path)

    image_paths = sorted(image_paths)

    if image_paths:
        return dataset_root, image_paths

    return possible_dirs[0], []


def box_iou_one_to_many(box: torch.Tensor, boxes: torch.Tensor, eps: float = 1e-7):
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


def nms(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    iou_threshold: float = 0.50,
):
    if boxes.numel() == 0:
        return torch.zeros((0,), dtype=torch.long, device=boxes.device)

    order = scores.argsort(descending=True)

    keep = []

    while order.numel() > 0:
        current = order[0]
        keep.append(current)

        if order.numel() == 1:
            break

        remaining = order[1:]

        ious = box_iou_one_to_many(
            box=boxes[current],
            boxes=boxes[remaining],
        )

        order = remaining[ious <= iou_threshold]

    return torch.stack(keep)


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

    This is safer than full class-agnostic NMS because it does not remove
    different-class boxes unless they look nearly identical.
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
            # This function is only for different-class duplicate boxes.
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


def letterbox_image(
    image: Image.Image,
    image_size: int,
):
    image = image.convert("RGB")

    original_width, original_height = image.size

    scale = min(
        image_size / original_width,
        image_size / original_height,
    )

    new_width = int(round(original_width * scale))
    new_height = int(round(original_height * scale))

    resized = image.resize((new_width, new_height), Image.BILINEAR)

    canvas = Image.new("RGB", (image_size, image_size), (114, 114, 114))

    pad_x = (image_size - new_width) // 2
    pad_y = (image_size - new_height) // 2

    canvas.paste(resized, (pad_x, pad_y))

    image_array = np.array(canvas).astype(np.float32) / 255.0

    tensor = torch.from_numpy(image_array).permute(2, 0, 1).unsqueeze(0)

    return tensor, scale, pad_x, pad_y


def map_boxes_back_to_original(
    boxes: torch.Tensor,
    scale: float,
    pad_x: int,
    pad_y: int,
    original_width: int,
    original_height: int,
):
    boxes = boxes.clone()

    boxes[:, [0, 2]] = (boxes[:, [0, 2]] - pad_x) / scale
    boxes[:, [1, 3]] = (boxes[:, [1, 3]] - pad_y) / scale

    boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(min=0, max=original_width)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(min=0, max=original_height)

    return boxes


def collect_predictions(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    confidence_threshold: float,
    nms_iou_threshold: float,
    max_detections: int,
    use_cross_class_duplicate_suppression: bool = True,
    duplicate_iou_threshold: float = 0.90,
    size_similarity_threshold: float = 0.95,
    center_distance_threshold: float = 0.05,
    nms_backend: str = "torchvision",
):
    if nms_backend not in {"python", "torchvision"}:
        raise ValueError(
            f"Unsupported NMS backend: {nms_backend!r}. "
            "Use 'python' or 'torchvision'."
        )

    if nms_backend == "torchvision":
        candidate_mask = scores >= confidence_threshold
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
            keep = class_scores >= confidence_threshold

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

    if order.numel() > max_detections:
        order = order[:max_detections]

    return final_boxes[order], final_scores[order], final_classes[order]


def draw_predictions(
    image: Image.Image,
    boxes: torch.Tensor,
    scores: torch.Tensor,
    class_ids: torch.Tensor,
):
    image = image.convert("RGB")
    draw = ImageDraw.Draw(image)

    try:
        font = ImageFont.truetype("arial.ttf", 14)
    except OSError:
        font = ImageFont.load_default()

    for box, score, class_id in zip(boxes, scores, class_ids):
        x1, y1, x2, y2 = box.tolist()
        class_id_int = int(class_id.item())

        label = f"{VOC_CLASSES[class_id_int]} {score.item():.2f}"

        draw.rectangle(
            [x1, y1, x2, y2],
            outline=(255, 0, 0),
            width=2,
        )

        text_box = draw.textbbox((x1, y1), label, font=font)
        text_width = text_box[2] - text_box[0]
        text_height = text_box[3] - text_box[1]

        text_y1 = max(y1 - text_height - 4, 0)

        draw.rectangle(
            [x1, text_y1, x1 + text_width + 4, text_y1 + text_height + 4],
            fill=(255, 0, 0),
        )

        draw.text(
            (x1 + 2, text_y1 + 2),
            label,
            fill=(255, 255, 255),
            font=font,
        )

    return image


@torch.inference_mode()
def predict_one_image(
    model,
    image_path: Path,
    output_path: Path,
    device,
    image_size: int,
    confidence_threshold: float,
    nms_iou_threshold: float,
    max_detections: int,
    use_cross_class_duplicate_suppression: bool,
    duplicate_iou_threshold: float,
    size_similarity_threshold: float,
    center_distance_threshold: float,
    precision: str,
    memory_format: str,
    nms_backend: str,
):
    image = Image.open(image_path).convert("RGB")
    original_width, original_height = image.size

    tensor, scale, pad_x, pad_y = letterbox_image(
        image=image,
        image_size=image_size,
    )

    tensor = move_images_to_device(
        images=tensor,
        device=device,
        memory_format=memory_format,
    )

    with inference_autocast(device=device, precision=precision):
        decoded = model(
            tensor,
            decode=True,
        )

    boxes = decoded["boxes"][0].float()
    scores = decoded["scores"][0].float()

    final_boxes, final_scores, final_classes = collect_predictions(
        boxes=boxes,
        scores=scores,
        confidence_threshold=confidence_threshold,
        nms_iou_threshold=nms_iou_threshold,
        max_detections=max_detections,
        use_cross_class_duplicate_suppression=use_cross_class_duplicate_suppression,
        duplicate_iou_threshold=duplicate_iou_threshold,
        size_similarity_threshold=size_similarity_threshold,
        center_distance_threshold=center_distance_threshold,
        nms_backend=nms_backend,
    )

    final_boxes = map_boxes_back_to_original(
        boxes=final_boxes,
        scale=scale,
        pad_x=pad_x,
        pad_y=pad_y,
        original_width=original_width,
        original_height=original_height,
    )

    result = draw_predictions(
        image=image,
        boxes=final_boxes.cpu(),
        scores=final_scores.cpu(),
        class_ids=final_classes.cpu(),
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.save(output_path)

    return len(final_boxes)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-root",
        type=str,
        default="data/processed/voc2007_2012_custom_voc8",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="runs/checkpoints_voc8_v08_768_mobilenetv3_distilled/best.pt",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="runs/predictions_voc8_v08_mobilenetv3_distilled",
    )
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--image-size", type=int, default=768)
    parser.add_argument("--conf", type=float, default=0.10)
    parser.add_argument("--nms-iou", type=float, default=0.50)
    parser.add_argument("--max-det", type=int, default=50)
    parser.add_argument("--num-images", type=int, default=12)
    parser.add_argument("--random-seed", type=int, default=None)
    parser.add_argument(
        "--precision",
        choices=["fp32", "fp16"],
        default="fp16",
    )
    parser.add_argument(
        "--memory-format",
        choices=["contiguous", "channels-last"],
        default="channels-last",
    )
    parser.add_argument(
        "--nms-backend",
        choices=["python", "torchvision"],
        default="torchvision",
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
    output_dir = Path(args.output_dir)

    if not dataset_root.is_absolute():
        dataset_root = project_root / dataset_root
    if not checkpoint_path.is_absolute():
        checkpoint_path = project_root / checkpoint_path
    if not output_dir.is_absolute():
        output_dir = project_root / output_dir

    image_size = args.image_size
    num_classes = len(VOC_CLASSES)
    use_cross_class_duplicate_suppression = (
        not args.disable_cross_class_duplicate_suppression
    )

    duplicate_iou_threshold = 0.90
    size_similarity_threshold = 0.95
    center_distance_threshold = 0.05

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.precision == "fp16" and device.type != "cuda":
        raise RuntimeError("FP16 prediction requires a CUDA device.")

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}\n"
            "Train the updated model first with:\n"
            "  python src\\train.py\n"
        )

    if not dataset_root.exists():
        raise FileNotFoundError(
            f"Dataset root not found: {dataset_root}\n"
            "Run this first:\n"
            "  python src\\build_voc8_experiment_split.py"
        )

    image_dir, image_paths = find_image_paths(
        dataset_root=dataset_root,
        split=args.split,
    )

    if not image_paths:
        raise FileNotFoundError(
            "No images found.\n"
            f"Tried:\n"
            f"  {dataset_root / args.split / 'images'}\n"
            f"  {dataset_root / 'images' / args.split}\n"
            f"  {dataset_root / args.split}\n"
        )

    if args.random_seed is not None:
        random.seed(args.random_seed)

    selected_paths = random.sample(
        image_paths,
        k=min(args.num_images, len(image_paths)),
    )

    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Expected experiment: {EXPECTED_EXPERIMENT_NAME}")
    print(f"Dataset root: {dataset_root}")
    print(f"Dataset split: {args.split}")
    print(f"Detected image directory: {image_dir}")
    print(f"Images found: {len(image_paths)}")
    print(f"Output directory: {output_dir}")
    print(f"Classes: {VOC_CLASSES}")
    print()
    print("Optimized prediction settings:")
    print(f"Precision: {args.precision}")
    print(f"Memory format: {args.memory_format}")
    print(f"NMS backend: {args.nms_backend}")
    print(f"torch.compile: {args.compile_model}")
    if args.compile_model:
        print(f"Compile mode: {args.compile_mode}")
    print(f"Confidence threshold: {args.conf}")
    print(f"Class-wise NMS IoU threshold: {args.nms_iou}")
    print(f"Max detections/image: {args.max_det}")
    print(
        "Cross-class duplicate suppression: "
        f"{use_cross_class_duplicate_suppression}"
    )
    print()

    model = TinyYOLOAnchorFree(
        num_classes=num_classes,
        image_size=image_size,
        reg_max=16,
        pretrained_backbone=False,
        use_auxiliary_heads=False,
    ).to(device)

    model_parameter_count = count_parameters(model)

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )
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
    print(f"Trainable params: {model_parameter_count:,}")
    print(f"Random seed: {args.random_seed}")
    print()

    for idx, image_path in enumerate(selected_paths):
        output_path = output_dir / f"prediction_{idx}.jpg"

        num_predictions = predict_one_image(
            model=model,
            image_path=image_path,
            output_path=output_path,
            device=device,
            image_size=image_size,
            confidence_threshold=args.conf,
            nms_iou_threshold=args.nms_iou,
            max_detections=args.max_det,
            use_cross_class_duplicate_suppression=(
                use_cross_class_duplicate_suppression
            ),
            duplicate_iou_threshold=duplicate_iou_threshold,
            size_similarity_threshold=size_similarity_threshold,
            center_distance_threshold=center_distance_threshold,
            precision=args.precision,
            memory_format=args.memory_format,
            nms_backend=args.nms_backend,
        )

        print(
            f"{idx + 1:02d}/{len(selected_paths)} | "
            f"{image_path.name} | "
            f"predictions {num_predictions} | "
            f"saved {output_path}"
        )

    print()
    print("Done.")


if __name__ == "__main__":
    main()