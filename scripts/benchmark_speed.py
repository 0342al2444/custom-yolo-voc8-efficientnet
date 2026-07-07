from pathlib import Path
import argparse
import time
import statistics

import torch
from torch.utils.data import DataLoader, Subset

from dataset import VOCDatasetYOLO, detection_collate_fn
from model import TinyYOLOAnchorFree, count_parameters
from eval import collect_predictions_per_image


EXPECTED_EXPERIMENT_NAME = "voc8_current_efficientnetb0_deeper_img960"


def validate_checkpoint_metadata(
    checkpoint: dict,
    checkpoint_path: Path,
    expected_experiment_name: str,
    expected_parameter_count: int,
) -> None:
    checkpoint_experiment = checkpoint.get("experiment_name")

    if checkpoint_experiment != expected_experiment_name:
        raise ValueError(
            "Checkpoint does not belong to the current experiment.\n"
            f"Checkpoint: {checkpoint_path}\n"
            f"Expected experiment: {expected_experiment_name!r}\n"
            f"Found experiment:    {checkpoint_experiment!r}"
        )

    checkpoint_parameter_count = checkpoint.get("model_parameter_count")

    if checkpoint_parameter_count is not None:
        checkpoint_parameter_count = int(checkpoint_parameter_count)

        if checkpoint_parameter_count != expected_parameter_count:
            raise ValueError(
                "Checkpoint parameter count does not match the current model.\n"
                f"Checkpoint: {checkpoint_path}\n"
                f"Expected params: {expected_parameter_count}\n"
                f"Found params:    {checkpoint_parameter_count}"
            )


def sync_if_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


@torch.no_grad()
def time_forward_only(
    model: torch.nn.Module,
    device: torch.device,
    image_size: int,
    batch_size: int,
    warmup: int,
    runs: int,
    decode: bool,
):
    x = torch.randn(batch_size, 3, image_size, image_size, device=device)

    for _ in range(warmup):
        _ = model(x, decode=decode)

    sync_if_cuda(device)

    times_ms = []

    for _ in range(runs):
        sync_if_cuda(device)
        start = time.perf_counter()

        _ = model(x, decode=decode)

        sync_if_cuda(device)
        end = time.perf_counter()

        times_ms.append((end - start) * 1000.0)

    return {
        "mean_ms_per_batch": statistics.mean(times_ms),
        "median_ms_per_batch": statistics.median(times_ms),
        "mean_ms_per_image": statistics.mean(times_ms) / batch_size,
        "median_ms_per_image": statistics.median(times_ms) / batch_size,
        "fps": 1000.0 / (statistics.mean(times_ms) / batch_size),
    }


@torch.no_grad()
def time_real_dataset(
    model: torch.nn.Module,
    device: torch.device,
    dataset_root: Path,
    split: str,
    image_size: int,
    batch_size: int,
    max_images: int,
    conf_threshold: float,
    nms_iou_threshold: float,
    max_detections_per_image: int,
):
    dataset = VOCDatasetYOLO(
        root_dir=dataset_root,
        split=split,
        image_size=image_size,
        augment=False,
    )

    total_available = len(dataset)

    if max_images > 0:
        use_images = min(max_images, total_available)
        dataset = Subset(dataset, list(range(use_images)))
    else:
        use_images = total_available

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
        collate_fn=detection_collate_fn,
    )

    batch_times_ms = []
    preprocess_transfer_times_ms = []
    forward_decode_times_ms = []
    postprocess_times_ms = []

    total_images = 0
    total_predictions_kept = 0

    # One warmup batch from the real loader.
    warm_batch = next(iter(loader))
    warm_images = warm_batch[0].to(device, non_blocking=True)
    _ = model(warm_images, decode=True)
    sync_if_cuda(device)

    for images, targets, image_paths in loader:
        batch_count = images.shape[0]
        total_images += batch_count

        start_pre = time.perf_counter()
        images = images.to(device, non_blocking=True)
        sync_if_cuda(device)
        end_pre = time.perf_counter()

        start_forward = time.perf_counter()
        decoded = model(images, decode=True)
        sync_if_cuda(device)
        end_forward = time.perf_counter()

        start_post = time.perf_counter()

        boxes_batch = decoded["boxes"]
        scores_batch = decoded["scores"]

        for image_index in range(batch_count):
            boxes, scores, class_ids = collect_predictions_per_image(
                boxes=boxes_batch[image_index],
                scores=scores_batch[image_index],
                conf_threshold=conf_threshold,
                nms_iou_threshold=nms_iou_threshold,
                max_detections_per_image=max_detections_per_image,
                use_cross_class_duplicate_suppression=True,
                duplicate_iou_threshold=0.90,
                size_similarity_threshold=0.95,
                center_distance_threshold=0.05,
            )
            total_predictions_kept += int(boxes.shape[0])

        sync_if_cuda(device)
        end_post = time.perf_counter()

        preprocess_transfer_times_ms.append((end_pre - start_pre) * 1000.0)
        forward_decode_times_ms.append((end_forward - start_forward) * 1000.0)
        postprocess_times_ms.append((end_post - start_post) * 1000.0)
        batch_times_ms.append((end_post - start_pre) * 1000.0)

    mean_total_batch = statistics.mean(batch_times_ms)
    mean_pre_batch = statistics.mean(preprocess_transfer_times_ms)
    mean_forward_batch = statistics.mean(forward_decode_times_ms)
    mean_post_batch = statistics.mean(postprocess_times_ms)

    return {
        "images_used": total_images,
        "available_images": total_available,
        "mean_total_ms_per_batch": mean_total_batch,
        "mean_preprocess_transfer_ms_per_batch": mean_pre_batch,
        "mean_forward_decode_ms_per_batch": mean_forward_batch,
        "mean_postprocess_ms_per_batch": mean_post_batch,
        "mean_total_ms_per_image": mean_total_batch / batch_size,
        "mean_preprocess_transfer_ms_per_image": mean_pre_batch / batch_size,
        "mean_forward_decode_ms_per_image": mean_forward_batch / batch_size,
        "mean_postprocess_ms_per_image": mean_post_batch / batch_size,
        "fps_end_to_end": 1000.0 / (mean_total_batch / batch_size),
        "average_predictions_kept_per_image": total_predictions_kept / max(total_images, 1),
    }


def print_result_block(title: str, result: dict) -> None:
    print()
    print("=" * 80)
    print(title)
    print("=" * 80)

    for key, value in result.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset-root", type=str, default="data/processed/voc2007_2012_custom_voc8")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--checkpoint", type=str, default="runs/checkpoints_voc8_current/best.pt")
    parser.add_argument("--image-size", type=int, default=960)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-images", type=int, default=200)
    parser.add_argument("--conf", type=float, default=0.10)
    parser.add_argument("--nms-iou", type=float, default=0.50)
    parser.add_argument("--max-det", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--runs", type=int, default=100)

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    project_root = Path(__file__).resolve().parents[1]
    dataset_root = Path(args.dataset_root)
    checkpoint_path = Path(args.checkpoint)

    if not dataset_root.is_absolute():
        dataset_root = project_root / dataset_root

    if not checkpoint_path.is_absolute():
        checkpoint_path = project_root / checkpoint_path

    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    print(f"Dataset root: {dataset_root}")
    print(f"Split: {args.split}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Image size: {args.image_size}")
    print(f"Batch size: {args.batch_size}")
    print(f"Max images for real dataset timing: {args.max_images}")
    print(f"Confidence threshold: {args.conf}")

    model = TinyYOLOAnchorFree(
        num_classes=8,
        image_size=args.image_size,
        pretrained_backbone=False,
        use_auxiliary_heads=True,
    ).to(device)

    parameter_count = count_parameters(model)

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    validate_checkpoint_metadata(
        checkpoint=checkpoint,
        checkpoint_path=checkpoint_path,
        expected_experiment_name=EXPECTED_EXPERIMENT_NAME,
        expected_parameter_count=parameter_count,
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print(f"Loaded epoch: {checkpoint.get('epoch', 'unknown')}")
    print(f"Checkpoint val loss: {checkpoint.get('val_loss', 'unknown')}")
    print(f"Trainable params: {parameter_count:,}")

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    raw_forward = time_forward_only(
        model=model,
        device=device,
        image_size=args.image_size,
        batch_size=args.batch_size,
        warmup=args.warmup,
        runs=args.runs,
        decode=False,
    )

    print_result_block(
        title="Dummy tensor speed: model forward only, no decode, no NMS",
        result=raw_forward,
    )

    forward_decode = time_forward_only(
        model=model,
        device=device,
        image_size=args.image_size,
        batch_size=args.batch_size,
        warmup=args.warmup,
        runs=args.runs,
        decode=True,
    )

    print_result_block(
        title="Dummy tensor speed: model forward + box decode, no NMS",
        result=forward_decode,
    )

    real_dataset_speed = time_real_dataset(
        model=model,
        device=device,
        dataset_root=dataset_root,
        split=args.split,
        image_size=args.image_size,
        batch_size=args.batch_size,
        max_images=args.max_images,
        conf_threshold=args.conf,
        nms_iou_threshold=args.nms_iou,
        max_detections_per_image=args.max_det,
    )

    print_result_block(
        title="Real dataset speed: image tensor transfer + forward decode + NMS",
        result=real_dataset_speed,
    )

    print()
    print("Done.")


if __name__ == "__main__":
    main()
