from pathlib import Path
import argparse
from contextlib import nullcontext
import csv
import statistics
import time

import torch
from torch.utils.data import DataLoader, Subset

from dataset import VOCDatasetYOLO, detection_collate_fn
from model import (
    TinyYOLOAnchorFree,
    count_parameters,
    load_inference_state_dict,
)
from eval import collect_predictions_per_image


EXPECTED_EXPERIMENT_NAME = "voc8_v08_768_mobilenetv3_distilled"


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
            f"Found experiment:    {checkpoint_experiment!r}"
        )

    checkpoint_parameter_count = checkpoint.get("inference_parameter_count")

    if checkpoint_parameter_count is not None:
        checkpoint_parameter_count = int(checkpoint_parameter_count)

        if checkpoint_parameter_count != expected_inference_parameter_count:
            raise ValueError(
                "Checkpoint parameter count does not match the current model.\n"
                f"Checkpoint: {checkpoint_path}\n"
                f"Expected inference params: {expected_inference_parameter_count}\n"
                f"Found params:    {checkpoint_parameter_count}"
            )


def sync_if_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


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


def summarize_timing_window(
    start_image: int,
    end_image: int,
    preprocess_transfer_times_ms: list[float],
    forward_decode_times_ms: list[float],
    postprocess_times_ms: list[float],
    total_times_ms: list[float],
    predictions_kept: int,
) -> dict:
    images_used = end_image - start_image + 1

    if images_used <= 0:
        raise ValueError("Timing window must contain at least one image.")

    total_pre_ms = sum(preprocess_transfer_times_ms)
    total_forward_ms = sum(forward_decode_times_ms)
    total_post_ms = sum(postprocess_times_ms)
    total_elapsed_ms = sum(total_times_ms)

    mean_total_ms_per_image = total_elapsed_ms / images_used

    return {
        "start_image": start_image,
        "end_image": end_image,
        "images": images_used,
        "mean_total_ms_per_batch": statistics.mean(total_times_ms),
        "mean_preprocess_transfer_ms_per_batch": statistics.mean(
            preprocess_transfer_times_ms
        ),
        "mean_forward_decode_ms_per_batch": statistics.mean(
            forward_decode_times_ms
        ),
        "mean_postprocess_ms_per_batch": statistics.mean(postprocess_times_ms),
        "mean_total_ms_per_image": mean_total_ms_per_image,
        "mean_preprocess_transfer_ms_per_image": total_pre_ms / images_used,
        "mean_forward_decode_ms_per_image": total_forward_ms / images_used,
        "mean_postprocess_ms_per_image": total_post_ms / images_used,
        "fps_end_to_end": 1000.0 / mean_total_ms_per_image,
        "average_predictions_kept_per_image": predictions_kept / images_used,
    }


def print_segment_line(segment_index: int, segment: dict) -> None:
    print(
        f"Segment {segment_index:02d} | "
        f"images {segment['start_image']}-{segment['end_image']} | "
        f"total {segment['mean_total_ms_per_image']:.4f} ms/img | "
        f"forward {segment['mean_forward_decode_ms_per_image']:.4f} | "
        f"post {segment['mean_postprocess_ms_per_image']:.4f} | "
        f"FPS {segment['fps_end_to_end']:.4f} | "
        f"pred/img {segment['average_predictions_kept_per_image']:.4f}"
    )


def save_segment_results_csv(
    csv_path: Path,
    segment_results: list[dict],
) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "segment",
        "start_image",
        "end_image",
        "images",
        "mean_total_ms_per_image",
        "mean_preprocess_transfer_ms_per_image",
        "mean_forward_decode_ms_per_image",
        "mean_postprocess_ms_per_image",
        "fps_end_to_end",
        "average_predictions_kept_per_image",
    ]

    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        for segment_index, segment in enumerate(segment_results, start=1):
            writer.writerow(
                {
                    "segment": segment_index,
                    "start_image": segment["start_image"],
                    "end_image": segment["end_image"],
                    "images": segment["images"],
                    "mean_total_ms_per_image": segment[
                        "mean_total_ms_per_image"
                    ],
                    "mean_preprocess_transfer_ms_per_image": segment[
                        "mean_preprocess_transfer_ms_per_image"
                    ],
                    "mean_forward_decode_ms_per_image": segment[
                        "mean_forward_decode_ms_per_image"
                    ],
                    "mean_postprocess_ms_per_image": segment[
                        "mean_postprocess_ms_per_image"
                    ],
                    "fps_end_to_end": segment["fps_end_to_end"],
                    "average_predictions_kept_per_image": segment[
                        "average_predictions_kept_per_image"
                    ],
                }
            )


@torch.inference_mode()
def time_forward_only(
    model: torch.nn.Module,
    device: torch.device,
    image_size: int,
    batch_size: int,
    warmup: int,
    runs: int,
    decode: bool,
    precision: str,
    memory_format: str,
):
    x = torch.randn(
        batch_size,
        3,
        image_size,
        image_size,
        device=device,
    )

    if memory_format == "channels-last":
        x = x.contiguous(memory_format=torch.channels_last)

    for _ in range(warmup):
        with inference_autocast(device=device, precision=precision):
            _ = model(x, decode=decode)

    sync_if_cuda(device)

    times_ms = []

    for _ in range(runs):
        sync_if_cuda(device)
        start = time.perf_counter()

        with inference_autocast(device=device, precision=precision):
            _ = model(x, decode=decode)

        sync_if_cuda(device)
        end = time.perf_counter()

        times_ms.append((end - start) * 1000.0)

    mean_ms_per_batch = statistics.mean(times_ms)
    median_ms_per_batch = statistics.median(times_ms)

    return {
        "mean_ms_per_batch": mean_ms_per_batch,
        "median_ms_per_batch": median_ms_per_batch,
        "mean_ms_per_image": mean_ms_per_batch / batch_size,
        "median_ms_per_image": median_ms_per_batch / batch_size,
        "fps": 1000.0 / (mean_ms_per_batch / batch_size),
    }


@torch.inference_mode()
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
    precision: str,
    memory_format: str,
    nms_backend: str,
    use_cross_class_duplicate_suppression: bool,
    report_every: int,
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

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
        collate_fn=detection_collate_fn,
    )

    preprocess_transfer_times_ms = []
    forward_decode_times_ms = []
    postprocess_times_ms = []
    total_times_ms = []

    segment_preprocess_transfer_times_ms = []
    segment_forward_decode_times_ms = []
    segment_postprocess_times_ms = []
    segment_total_times_ms = []

    segment_results = []

    total_images = 0
    total_predictions_kept = 0

    segment_start_image = 1
    segment_images = 0
    segment_predictions_kept = 0

    warm_images = next(iter(loader))[0]
    warm_images = move_images_to_device(
        images=warm_images,
        device=device,
        memory_format=memory_format,
    )

    with inference_autocast(device=device, precision=precision):
        _ = model(warm_images, decode=True)
    sync_if_cuda(device)

    print()
    print("Per-segment timing during the real dataset benchmark:")

    for images, targets, image_paths in loader:
        batch_count = images.shape[0]
        total_images += batch_count
        segment_images += batch_count

        start_pre = time.perf_counter()
        images = move_images_to_device(
            images=images,
            device=device,
            memory_format=memory_format,
        )
        sync_if_cuda(device)
        end_pre = time.perf_counter()

        start_forward = time.perf_counter()
        with inference_autocast(device=device, precision=precision):
            decoded = model(images, decode=True)
        sync_if_cuda(device)
        end_forward = time.perf_counter()

        start_post = time.perf_counter()

        boxes_batch = decoded["boxes"].float()
        scores_batch = decoded["scores"].float()

        batch_predictions_kept = 0

        for image_index in range(batch_count):
            boxes, scores, class_ids = collect_predictions_per_image(
                boxes=boxes_batch[image_index],
                scores=scores_batch[image_index],
                conf_threshold=conf_threshold,
                nms_iou_threshold=nms_iou_threshold,
                max_detections_per_image=max_detections_per_image,
                use_cross_class_duplicate_suppression=(
                    use_cross_class_duplicate_suppression
                ),
                duplicate_iou_threshold=0.90,
                size_similarity_threshold=0.95,
                center_distance_threshold=0.05,
                nms_backend=nms_backend,
            )
            batch_predictions_kept += int(boxes.shape[0])

        total_predictions_kept += batch_predictions_kept
        segment_predictions_kept += batch_predictions_kept

        sync_if_cuda(device)
        end_post = time.perf_counter()

        preprocess_ms = (end_pre - start_pre) * 1000.0
        forward_ms = (end_forward - start_forward) * 1000.0
        postprocess_ms = (end_post - start_post) * 1000.0
        total_ms = (end_post - start_pre) * 1000.0

        preprocess_transfer_times_ms.append(preprocess_ms)
        forward_decode_times_ms.append(forward_ms)
        postprocess_times_ms.append(postprocess_ms)
        total_times_ms.append(total_ms)

        segment_preprocess_transfer_times_ms.append(preprocess_ms)
        segment_forward_decode_times_ms.append(forward_ms)
        segment_postprocess_times_ms.append(postprocess_ms)
        segment_total_times_ms.append(total_ms)

        reached_report_boundary = report_every > 0 and segment_images >= report_every
        reached_dataset_end = total_images >= len(dataset)

        if reached_report_boundary or reached_dataset_end:
            segment = summarize_timing_window(
                start_image=segment_start_image,
                end_image=total_images,
                preprocess_transfer_times_ms=(
                    segment_preprocess_transfer_times_ms
                ),
                forward_decode_times_ms=segment_forward_decode_times_ms,
                postprocess_times_ms=segment_postprocess_times_ms,
                total_times_ms=segment_total_times_ms,
                predictions_kept=segment_predictions_kept,
            )
            segment_results.append(segment)
            print_segment_line(len(segment_results), segment)

            segment_start_image = total_images + 1
            segment_images = 0
            segment_predictions_kept = 0
            segment_preprocess_transfer_times_ms = []
            segment_forward_decode_times_ms = []
            segment_postprocess_times_ms = []
            segment_total_times_ms = []

    total_pre_ms = sum(preprocess_transfer_times_ms)
    total_forward_ms = sum(forward_decode_times_ms)
    total_post_ms = sum(postprocess_times_ms)
    total_elapsed_ms = sum(total_times_ms)

    overall_result = {
        "images_used": total_images,
        "available_images": total_available,
        "mean_total_ms_per_batch": statistics.mean(total_times_ms),
        "mean_preprocess_transfer_ms_per_batch": statistics.mean(
            preprocess_transfer_times_ms
        ),
        "mean_forward_decode_ms_per_batch": statistics.mean(
            forward_decode_times_ms
        ),
        "mean_postprocess_ms_per_batch": statistics.mean(postprocess_times_ms),
        "mean_total_ms_per_image": total_elapsed_ms / max(total_images, 1),
        "mean_preprocess_transfer_ms_per_image": (
            total_pre_ms / max(total_images, 1)
        ),
        "mean_forward_decode_ms_per_image": (
            total_forward_ms / max(total_images, 1)
        ),
        "mean_postprocess_ms_per_image": total_post_ms / max(total_images, 1),
        "fps_end_to_end": 1000.0 / (total_elapsed_ms / max(total_images, 1)),
        "average_predictions_kept_per_image": (
            total_predictions_kept / max(total_images, 1)
        ),
    }

    return overall_result, segment_results


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


def print_segment_table(segment_results: list[dict]) -> None:
    if not segment_results:
        return

    print()
    print("=" * 106)
    print("Sustained real-dataset timing by image segment")
    print("=" * 106)
    print(
        f"{'Segment':>7}  {'Images':>13}  {'Transfer':>10}  "
        f"{'Forward':>10}  {'Post':>10}  {'Total':>10}  "
        f"{'FPS':>9}  {'Pred/img':>9}"
    )
    print("-" * 106)

    for segment_index, segment in enumerate(segment_results, start=1):
        image_range = f"{segment['start_image']}-{segment['end_image']}"

        print(
            f"{segment_index:>7}  "
            f"{image_range:>13}  "
            f"{segment['mean_preprocess_transfer_ms_per_image']:>10.4f}  "
            f"{segment['mean_forward_decode_ms_per_image']:>10.4f}  "
            f"{segment['mean_postprocess_ms_per_image']:>10.4f}  "
            f"{segment['mean_total_ms_per_image']:>10.4f}  "
            f"{segment['fps_end_to_end']:>9.4f}  "
            f"{segment['average_predictions_kept_per_image']:>9.4f}"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-root",
        type=str,
        default="data/processed/voc2007_2012_custom_voc8",
    )
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="runs/checkpoints_voc8_v08_768_mobilenetv3_distilled/best.pt",
    )
    parser.add_argument("--image-size", type=int, default=768)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-images", type=int, default=200)
    parser.add_argument("--conf", type=float, default=0.10)
    parser.add_argument("--nms-iou", type=float, default=0.50)
    parser.add_argument("--max-det", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--runs", type=int, default=200)
    parser.add_argument(
        "--report-every",
        type=int,
        default=500,
        help=(
            "Print one sustained timing segment after approximately this many "
            "images. Use 0 to disable intermediate segmentation."
        ),
    )
    parser.add_argument(
        "--segment-csv",
        type=str,
        default="",
        help=(
            "Optional CSV path for per-segment timing results. Relative paths "
            "are resolved from the project root."
        ),
    )
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

    if args.report_every < 0:
        raise ValueError("--report-every must be 0 or a positive integer.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.precision == "fp16" and device.type != "cuda":
        raise RuntimeError("FP16 benchmarking requires a CUDA device.")

    project_root = Path(__file__).resolve().parents[1]
    dataset_root = Path(args.dataset_root)
    checkpoint_path = Path(args.checkpoint)

    if not dataset_root.is_absolute():
        dataset_root = project_root / dataset_root

    if not checkpoint_path.is_absolute():
        checkpoint_path = project_root / checkpoint_path

    segment_csv_path = None

    if args.segment_csv:
        segment_csv_path = Path(args.segment_csv)

        if not segment_csv_path.is_absolute():
            segment_csv_path = project_root / segment_csv_path

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    use_cross_class_duplicate_suppression = (
        not args.disable_cross_class_duplicate_suppression
    )

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
    print(f"Precision: {args.precision}")
    print(f"Memory format: {args.memory_format}")
    print(f"NMS backend: {args.nms_backend}")
    print(f"Report every: {args.report_every} images")
    if segment_csv_path is not None:
        print(f"Segment CSV: {segment_csv_path}")
    print(
        "Cross-class duplicate suppression: "
        f"{use_cross_class_duplicate_suppression}"
    )
    print(f"torch.compile: {args.compile_model}")
    if args.compile_model:
        print(f"Compile mode: {args.compile_mode}")

    model = TinyYOLOAnchorFree(
        num_classes=8,
        image_size=args.image_size,
        reg_max=16,
        pretrained_backbone=False,
        use_auxiliary_heads=False,
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
        expected_inference_parameter_count=parameter_count,
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

    print(f"Loaded epoch: {checkpoint.get('epoch', 'unknown')}")
    print(f"Checkpoint val loss: {checkpoint.get('val_loss', 'unknown')}")
    print(f"Trainable params: {parameter_count:,}")

    raw_forward = time_forward_only(
        model=model,
        device=device,
        image_size=args.image_size,
        batch_size=args.batch_size,
        warmup=args.warmup,
        runs=args.runs,
        decode=False,
        precision=args.precision,
        memory_format=args.memory_format,
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
        precision=args.precision,
        memory_format=args.memory_format,
    )

    print_result_block(
        title="Dummy tensor speed: model forward + box decode, no NMS",
        result=forward_decode,
    )

    real_dataset_speed, segment_results = time_real_dataset(
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
        precision=args.precision,
        memory_format=args.memory_format,
        nms_backend=args.nms_backend,
        use_cross_class_duplicate_suppression=(
            use_cross_class_duplicate_suppression
        ),
        report_every=args.report_every,
    )

    print_result_block(
        title=(
            "Real dataset speed: tensor transfer + forward decode + "
            f"{args.nms_backend} NMS"
        ),
        result=real_dataset_speed,
    )

    print_segment_table(segment_results)

    if segment_csv_path is not None:
        save_segment_results_csv(
            csv_path=segment_csv_path,
            segment_results=segment_results,
        )
        print()
        print(f"Saved segment timing CSV: {segment_csv_path}")

    print()
    print("Done.")


if __name__ == "__main__":
    main()
