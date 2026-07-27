from pathlib import Path
import argparse

import torch

from model import (
    TinyYOLOAnchorFree,
    count_parameters,
    load_inference_state_dict,
)


EXPECTED_EXPERIMENT_NAME = "voc8_v08_768_mobilenetv3_distilled"


def validate_checkpoint(
    checkpoint: dict,
    checkpoint_path: Path,
    expected_inference_parameter_count: int,
) -> None:
    experiment_name = checkpoint.get("experiment_name")

    if experiment_name != EXPECTED_EXPERIMENT_NAME:
        raise ValueError(
            "Checkpoint does not belong to the v0.8 MobileNetV3 distilled experiment.\n"
            f"Checkpoint: {checkpoint_path}\n"
            f"Expected:   {EXPECTED_EXPERIMENT_NAME!r}\n"
            f"Found:      {experiment_name!r}"
        )

    stored_count = checkpoint.get("inference_parameter_count")

    if stored_count is not None and int(stored_count) != expected_inference_parameter_count:
        raise ValueError(
            "Checkpoint inference parameter count does not match the model.\n"
            f"Expected: {expected_inference_parameter_count}\n"
            f"Found:    {int(stored_count)}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="runs/checkpoints_voc8_v08_768_mobilenetv3_distilled/best.pt",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=(
            "runs/checkpoints_voc8_v08_768_mobilenetv3_distilled/"
            "best_deployment.pt"
        ),
    )
    parser.add_argument("--image-size", type=int, default=768)
    parser.add_argument("--num-classes", type=int, default=8)
    parser.add_argument("--reg-max", type=int, default=16)
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    checkpoint_path = (project_root / args.checkpoint).resolve()
    output_path = (project_root / args.output).resolve()

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    model = TinyYOLOAnchorFree(
        num_classes=args.num_classes,
        image_size=args.image_size,
        reg_max=args.reg_max,
        pretrained_backbone=False,
        use_auxiliary_heads=False,
    )

    inference_parameter_count = count_parameters(model)

    validate_checkpoint(
        checkpoint=checkpoint,
        checkpoint_path=checkpoint_path,
        expected_inference_parameter_count=inference_parameter_count,
    )

    source_state = checkpoint.get("model_state_dict")

    if source_state is None:
        raise KeyError("Checkpoint does not contain model_state_dict.")

    load_inference_state_dict(model, source_state)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "checkpoint_type": "inference_only",
            "backbone": "MobileNetV3-Large",
            "distilled_from": "voc8_v06_768_no_p2_regmax16_depth_slim",
            "experiment_name": EXPECTED_EXPERIMENT_NAME,
            "experiment_description": checkpoint.get("experiment_description"),
            "source_checkpoint": str(checkpoint_path),
            "source_epoch": checkpoint.get("epoch"),
            "source_val_loss": checkpoint.get("val_loss"),
            "image_size": args.image_size,
            "num_classes": args.num_classes,
            "reg_max": args.reg_max,
            "strides": [8, 16, 32],
            "neck_channels": [56, 72, 112],
            "fusion_channels": 48,
            "neck_depths": {
                "fpn3": 3,
                "fpn4": 2,
                "pan4": 2,
                "pan5": 2,
            },
            "fusion_refine": {
                "n3": True,
                "n4": False,
                "n5": False,
            },
            "inference_parameter_count": inference_parameter_count,
            "model_parameter_count": inference_parameter_count,
            "model_state_dict": model.state_dict(),
        },
        output_path,
    )

    source_size_mb = checkpoint_path.stat().st_size / (1024**2)
    output_size_mb = output_path.stat().st_size / (1024**2)
    reduction_percent = 100.0 * (1.0 - output_size_mb / source_size_mb)

    print(f"Source checkpoint:     {checkpoint_path}")
    print(f"Deployment checkpoint: {output_path}")
    print(f"Inference parameters:  {inference_parameter_count:,}")
    print(f"Source size:           {source_size_mb:.2f} MB")
    print(f"Deployment size:       {output_size_mb:.2f} MB")
    print(f"File-size reduction:   {reduction_percent:.1f}%")


if __name__ == "__main__":
    main()
