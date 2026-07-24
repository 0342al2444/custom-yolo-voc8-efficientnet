from pathlib import Path
import csv
import time
import copy
import math

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW

from dataset import VOCDatasetYOLO, detection_collate_fn, VOC_CLASSES
from model import TinyYOLOAnchorFree, count_parameters, count_inference_parameters
from teacher_model import (
    YOLOv06Teacher,
    V06_EXPERIMENT_NAME,
    load_v06_teacher_checkpoint,
)
from distillation import DetectionDistillationLoss
from loss import YOLOAnchorFreeLoss


EXPERIMENT_NAME = "voc8_v08_768_mobilenetv3_distilled"
EXPERIMENT_DESCRIPTION = (
    "v0.8 VOC8 compact distilled detector: pretrained MobileNetV3-Large "
    "backbone, 56/72/112 FPN-PAN neck, Gold-YOLO-lite fusion width 48, "
    "48-channel minimum detection-head width, 768x768 input, no P2 head, "
    "reg_max=16, and detection-specific knowledge distillation from the "
    "frozen v0.6 EfficientNet-B0 teacher."
)


POSITIVE_DIAGNOSTIC_KEYS = [
    "positive_stride_8",
    "positive_stride_16",
    "positive_stride_32",
    "positive_small",
    "positive_medium",
    "positive_large",
]


def loss_value(loss_dict: dict, key: str) -> float:
    value = loss_dict.get(key)

    if value is None:
        return 0.0

    if torch.is_tensor(value):
        return float(value.detach().item())

    return float(value)



def add_positive_diagnostics_from_loss(
    running: dict,
    loss_dict: dict,
) -> None:
    running["positive_stride_8"] += loss_value(loss_dict, "num_positive_stride_8")
    running["positive_stride_16"] += loss_value(loss_dict, "num_positive_stride_16")
    running["positive_stride_32"] += loss_value(loss_dict, "num_positive_stride_32")
    running["positive_small"] += loss_value(loss_dict, "num_positive_small")
    running["positive_medium"] += loss_value(loss_dict, "num_positive_medium")
    running["positive_large"] += loss_value(loss_dict, "num_positive_large")


def average_positive_diagnostics(
    running: dict,
    num_batches: int,
) -> dict:
    return {
        key: running[key] / max(num_batches, 1)
        for key in POSITIVE_DIAGNOSTIC_KEYS
    }



class ModelEMA:
    def __init__(self, model, decay: float = 0.9998):
        self.ema = copy.deepcopy(model).eval()
        self.decay = decay
        self.updates = 0

        for param in self.ema.parameters():
            param.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        self.updates += 1

        decay = self.decay * (1.0 - math.exp(-self.updates / 2000.0))

        model_state = model.state_dict()
        ema_state = self.ema.state_dict()

        for key, ema_value in ema_state.items():
            model_value = model_state[key].detach()

            if ema_value.dtype.is_floating_point:
                ema_value.mul_(decay).add_(model_value, alpha=1.0 - decay)
            else:
                ema_value.copy_(model_value)


def adjust_learning_rate(
    optimizer,
    base_lr: float,
    min_lr: float,
    epoch: int,
    batch_idx: int,
    num_batches: int,
    num_epochs: int,
    warmup_epochs: int,
):
    current_step = (epoch - 1) * num_batches + batch_idx
    total_steps = num_epochs * num_batches
    warmup_steps = warmup_epochs * num_batches

    if current_step <= warmup_steps:
        warmup_start_factor = 0.05
        progress = current_step / max(warmup_steps, 1)
        lr = base_lr * (warmup_start_factor + progress * (1.0 - warmup_start_factor))
    else:
        progress = (current_step - warmup_steps) / max(total_steps - warmup_steps, 1)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        lr = min_lr + (base_lr - min_lr) * cosine

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

    return lr


def save_checkpoint(
    checkpoint_path: Path,
    model,
    ema,
    distiller,
    optimizer,
    scaler,
    epoch: int,
    train_loss: float,
    val_loss: float,
    best_val_loss: float,
    experiment_name: str,
    experiment_description: str,
    model_parameter_count: int,
):
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "experiment_name": experiment_name,
            "experiment_description": experiment_description,
            "model_parameter_count": model_parameter_count,
            "inference_parameter_count": count_inference_parameters(model),
            "epoch": epoch,
            "model_state_dict": ema.ema.state_dict(),
            "raw_model_state_dict": model.state_dict(),
            "distiller_state_dict": distiller.state_dict(),
            "distiller_parameter_count": count_parameters(distiller),
            "optimizer_state_dict": optimizer.state_dict(),
            "teacher_experiment_name": V06_EXPERIMENT_NAME,
            "scaler_state_dict": scaler.state_dict(),
            "ema_updates": ema.updates,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "best_val_loss": best_val_loss,
        },
        checkpoint_path,
    )


def inspect_checkpoint_for_resume(
    checkpoint_path: Path,
    expected_experiment_name: str,
    expected_parameter_count: int,
):
    """
    Checks whether a checkpoint belongs to the current experiment.

    This prevents a new experiment from accidentally continuing from an older
    experiment checkpoint, while still allowing the same experiment to resume
    tomorrow from its own latest.pt.
    """

    if not checkpoint_path.exists():
        return False, "checkpoint does not exist", None

    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
    except Exception as error:
        return False, f"could not read checkpoint: {error}", None

    checkpoint_experiment = checkpoint.get("experiment_name")

    if checkpoint_experiment != expected_experiment_name:
        return (
            False,
            "checkpoint belongs to a different or older experiment "
            f"({checkpoint_experiment!r})",
            checkpoint,
        )

    checkpoint_parameter_count = checkpoint.get("model_parameter_count")

    if checkpoint_parameter_count is not None:
        checkpoint_parameter_count = int(checkpoint_parameter_count)

        if checkpoint_parameter_count != expected_parameter_count:
            return (
                False,
                "checkpoint parameter count does not match current model "
                f"({checkpoint_parameter_count} vs {expected_parameter_count})",
                checkpoint,
            )

    teacher_experiment = checkpoint.get("teacher_experiment_name")
    if teacher_experiment is not None and teacher_experiment != V06_EXPERIMENT_NAME:
        return (
            False,
            "checkpoint teacher metadata does not match v0.6 "
            f"({teacher_experiment!r})",
            checkpoint,
        )

    return True, "checkpoint matches current experiment", checkpoint


def remove_stale_training_files(
    paths: list[Path],
    reason: str,
):
    print("Existing checkpoint/history files are not safe for this experiment.")
    print(f"Reason: {reason}")
    print("Starting this version from epoch 1.")

    for path in paths:
        if path.exists():
            path.unlink()
            print(f"Removed stale file: {path}")

    print()


def load_training_checkpoint(
    checkpoint_path: Path,
    model,
    ema,
    distiller,
    optimizer,
    scaler,
    device,
    expected_experiment_name: str,
    expected_parameter_count: int,
):
    checkpoint = torch.load(checkpoint_path, map_location=device)

    checkpoint_experiment = checkpoint.get("experiment_name")

    if checkpoint_experiment != expected_experiment_name:
        raise ValueError(
            "Refusing to resume from a checkpoint that belongs to a different "
            f"experiment. Expected {expected_experiment_name!r}, "
            f"got {checkpoint_experiment!r}."
        )

    checkpoint_parameter_count = checkpoint.get("model_parameter_count")

    if checkpoint_parameter_count is not None:
        checkpoint_parameter_count = int(checkpoint_parameter_count)

        if checkpoint_parameter_count != expected_parameter_count:
            raise ValueError(
                "Refusing to resume because checkpoint parameter count does not "
                f"match current model. Expected {expected_parameter_count}, "
                f"got {checkpoint_parameter_count}."
            )

    raw_model_state = checkpoint.get("raw_model_state_dict")
    ema_model_state = checkpoint.get("model_state_dict")

    if raw_model_state is not None:
        model.load_state_dict(raw_model_state)
    elif ema_model_state is not None:
        # Backward-compatible fallback for older checkpoints.
        model.load_state_dict(ema_model_state)
    else:
        raise KeyError(
            "Checkpoint does not contain raw_model_state_dict or model_state_dict."
        )

    if ema_model_state is not None:
        ema.ema.load_state_dict(ema_model_state)
    else:
        ema.ema.load_state_dict(model.state_dict())

    distiller_state = checkpoint.get("distiller_state_dict")
    if distiller_state is None:
        raise KeyError("Checkpoint does not contain distiller_state_dict.")
    distiller.load_state_dict(distiller_state)

    optimizer_state = checkpoint.get("optimizer_state_dict")
    if optimizer_state is not None:
        optimizer.load_state_dict(optimizer_state)

    scaler_state = checkpoint.get("scaler_state_dict")
    if scaler_state is not None:
        scaler.load_state_dict(scaler_state)

    ema.updates = int(checkpoint.get("ema_updates", 0))

    completed_epoch = int(checkpoint.get("epoch", 0))
    next_epoch = completed_epoch + 1

    best_val_loss = float(
        checkpoint.get(
            "best_val_loss",
            checkpoint.get("val_loss", float("inf")),
        )
    )

    return checkpoint, next_epoch, best_val_loss


def append_history(
    history_path: Path,
    epoch: int,
    learning_rate: float,
    train_metrics: dict,
    val_metrics: dict,
    mosaic_prob: float,
):
    history_path.parent.mkdir(parents=True, exist_ok=True)

    file_exists = history_path.exists()

    with history_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        if not file_exists:
            writer.writerow(
                [
                    "epoch",
                    "learning_rate",
                    "mosaic_prob",
                    "train_total_loss",
                    "train_main_loss",
                    "train_aux_loss",
                    "train_distillation_scale",
                    "train_distillation_loss",
                    "train_distill_feature_loss",
                    "train_distill_classification_loss",
                    "train_distill_objectness_loss",
                    "train_distill_dfl_loss",
                    "train_cls_loss",
                    "train_obj_loss",
                    "train_box_loss",
                    "train_dfl_loss",
                    "train_positive_points",
                    "train_positive_stride_8",
                    "train_positive_stride_16",
                    "train_positive_stride_32",
                    "train_positive_small",
                    "train_positive_medium",
                    "train_positive_large",
                    "val_loss",
                    "val_cls_loss",
                    "val_obj_loss",
                    "val_box_loss",
                    "val_dfl_loss",
                    "val_positive_points",
                    "val_positive_stride_8",
                    "val_positive_stride_16",
                    "val_positive_stride_32",
                    "val_positive_small",
                    "val_positive_medium",
                    "val_positive_large",
                ]
            )

        writer.writerow(
            [
                epoch,
                learning_rate,
                mosaic_prob,
                train_metrics["loss"],
                train_metrics["main_loss"],
                train_metrics["aux_loss"],
                train_metrics["distillation_scale"],
                train_metrics["distillation_loss"],
                train_metrics["distill_feature_loss"],
                train_metrics["distill_classification_loss"],
                train_metrics["distill_objectness_loss"],
                train_metrics["distill_dfl_loss"],
                train_metrics["cls_loss"],
                train_metrics["obj_loss"],
                train_metrics["box_loss"],
                train_metrics["dfl_loss"],
                train_metrics["positive_points"],
                train_metrics["positive_stride_8"],
                train_metrics["positive_stride_16"],
                train_metrics["positive_stride_32"],
                train_metrics["positive_small"],
                train_metrics["positive_medium"],
                train_metrics["positive_large"],
                val_metrics["loss"],
                val_metrics["cls_loss"],
                val_metrics["obj_loss"],
                val_metrics["box_loss"],
                val_metrics["dfl_loss"],
                val_metrics["positive_points"],
                val_metrics["positive_stride_8"],
                val_metrics["positive_stride_16"],
                val_metrics["positive_stride_32"],
                val_metrics["positive_small"],
                val_metrics["positive_medium"],
                val_metrics["positive_large"],
            ]
        )


def distillation_schedule(epoch: int, num_epochs: int) -> float:
    """Warm up without distillation, use full KD, then taper for GT fitting."""

    if epoch <= 2:
        return 0.0

    if epoch >= max(3, num_epochs - 2):
        return 0.5

    return 1.0


def train_one_epoch(
    model,
    teacher,
    distiller,
    ema,
    criterion,
    loader,
    optimizer,
    scaler,
    device,
    epoch: int,
    num_epochs: int,
    base_lr: float,
    min_lr: float,
    warmup_epochs: int,
    aux_loss_weight: float,
    log_interval: int = 25,
):
    model.train()
    teacher.eval()
    distiller.train()

    kd_scale = distillation_schedule(epoch=epoch, num_epochs=num_epochs)

    running_total_loss = 0.0
    running_main_loss = 0.0
    running_aux_loss = 0.0
    running_distillation_loss = 0.0
    running_distill_feature_loss = 0.0
    running_distill_classification_loss = 0.0
    running_distill_objectness_loss = 0.0
    running_distill_dfl_loss = 0.0
    running_cls_loss = 0.0
    running_obj_loss = 0.0
    running_box_loss = 0.0
    running_dfl_loss = 0.0
    running_positive_points = 0
    running_positive_diagnostics = {
        key: 0.0
        for key in POSITIVE_DIAGNOSTIC_KEYS
    }

    start_time = time.time()
    last_lr = optimizer.param_groups[0]["lr"]

    for batch_idx, (images, targets, image_paths) in enumerate(loader, start=1):
        last_lr = adjust_learning_rate(
            optimizer=optimizer,
            base_lr=base_lr,
            min_lr=min_lr,
            epoch=epoch,
            batch_idx=batch_idx,
            num_batches=len(loader),
            num_epochs=num_epochs,
            warmup_epochs=warmup_epochs,
        )

        images = images.to(device, non_blocking=True)
        targets = [target.to(device, non_blocking=True) for target in targets]

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
            student_result = model(
                images,
                decode=False,
                return_aux=True,
                return_features=(kd_scale > 0.0),
            )

            main_loss_dict = criterion(student_result["main"], targets)
            aux_loss_dict = criterion(student_result["aux"], targets)

            main_loss = main_loss_dict["loss"]
            aux_loss = aux_loss_dict["loss"]

            if kd_scale > 0.0:
                with torch.no_grad():
                    teacher_result = teacher(images, return_features=True)

                distillation_dict = distiller(
                    student_result=student_result,
                    teacher_result=teacher_result,
                    targets=targets,
                )
                distillation_loss = distillation_dict["loss"]
            else:
                zero = main_loss * 0.0
                distillation_dict = {
                    "loss": zero,
                    "feature_loss": zero,
                    "classification_loss": zero,
                    "objectness_loss": zero,
                    "dfl_loss": zero,
                }
                distillation_loss = zero

            loss = (
                main_loss
                + aux_loss_weight * aux_loss
                + kd_scale * distillation_loss
            )

        scaler.scale(loss).backward()

        scaler.unscale_(optimizer)
        trainable_parameters = list(model.parameters()) + list(distiller.parameters())
        torch.nn.utils.clip_grad_norm_(trainable_parameters, max_norm=10.0)

        scaler.step(optimizer)
        scaler.update()

        ema.update(model)

        total_loss_value = loss.item()
        main_loss_value = main_loss.item()
        aux_loss_value = aux_loss.item()
        distillation_loss_value = distillation_loss.item()
        distill_feature_value = distillation_dict["feature_loss"].item()
        distill_classification_value = distillation_dict["classification_loss"].item()
        distill_objectness_value = distillation_dict["objectness_loss"].item()
        distill_dfl_value = distillation_dict["dfl_loss"].item()
        cls_loss_value = main_loss_dict["cls_loss"].item()
        obj_loss_value = main_loss_dict["obj_loss"].item()
        box_loss_value = main_loss_dict["box_loss"].item()
        dfl_loss_value = main_loss_dict["dfl_loss"].item()
        positive_points = int(main_loss_dict["num_positive_points"].item())

        add_positive_diagnostics_from_loss(
            running=running_positive_diagnostics,
            loss_dict=main_loss_dict,
        )

        running_total_loss += total_loss_value
        running_main_loss += main_loss_value
        running_aux_loss += aux_loss_value
        running_distillation_loss += distillation_loss_value
        running_distill_feature_loss += distill_feature_value
        running_distill_classification_loss += distill_classification_value
        running_distill_objectness_loss += distill_objectness_value
        running_distill_dfl_loss += distill_dfl_value
        running_cls_loss += cls_loss_value
        running_obj_loss += obj_loss_value
        running_box_loss += box_loss_value
        running_dfl_loss += dfl_loss_value
        running_positive_points += positive_points

        if batch_idx % log_interval == 0 or batch_idx == 1:
            elapsed = time.time() - start_time

            avg_total_loss = running_total_loss / batch_idx
            avg_main_loss = running_main_loss / batch_idx
            avg_aux_loss = running_aux_loss / batch_idx
            avg_kd_loss = running_distillation_loss / batch_idx
            avg_cls_loss = running_cls_loss / batch_idx
            avg_obj_loss = running_obj_loss / batch_idx
            avg_box_loss = running_box_loss / batch_idx
            avg_dfl_loss = running_dfl_loss / batch_idx
            avg_pos = running_positive_points / batch_idx
            avg_diag = average_positive_diagnostics(
                running=running_positive_diagnostics,
                num_batches=batch_idx,
            )

            print(
                f"Epoch {epoch} | "
                f"Batch {batch_idx}/{len(loader)} | "
                f"lr {last_lr:.6f} | "
                f"loss {avg_total_loss:.4f} | "
                f"main {avg_main_loss:.4f} | "
                f"aux {avg_aux_loss:.4f} | "
                f"kd {avg_kd_loss:.4f} x {kd_scale:.2f} | "
                f"cls {avg_cls_loss:.4f} | "
                f"obj {avg_obj_loss:.4f} | "
                f"box {avg_box_loss:.4f} | "
                f"dfl {avg_dfl_loss:.4f} | "
                f"pos {avg_pos:.1f} | "
                f"s8/s16/s32 "
                f"{avg_diag['positive_stride_8']:.1f}/"
                f"{avg_diag['positive_stride_16']:.1f}/"
                f"{avg_diag['positive_stride_32']:.1f} | "
                f"sm/md/lg "
                f"{avg_diag['positive_small']:.1f}/"
                f"{avg_diag['positive_medium']:.1f}/"
                f"{avg_diag['positive_large']:.1f} | "
                f"time {elapsed:.1f}s"
            )

    avg_positive_diagnostics = average_positive_diagnostics(
        running=running_positive_diagnostics,
        num_batches=len(loader),
    )

    return {
        "loss": running_total_loss / len(loader),
        "main_loss": running_main_loss / len(loader),
        "aux_loss": running_aux_loss / len(loader),
        "distillation_scale": kd_scale,
        "distillation_loss": running_distillation_loss / len(loader),
        "distill_feature_loss": running_distill_feature_loss / len(loader),
        "distill_classification_loss": (
            running_distill_classification_loss / len(loader)
        ),
        "distill_objectness_loss": running_distill_objectness_loss / len(loader),
        "distill_dfl_loss": running_distill_dfl_loss / len(loader),
        "cls_loss": running_cls_loss / len(loader),
        "obj_loss": running_obj_loss / len(loader),
        "box_loss": running_box_loss / len(loader),
        "dfl_loss": running_dfl_loss / len(loader),
        "positive_points": running_positive_points / len(loader),
        "positive_stride_8": avg_positive_diagnostics["positive_stride_8"],
        "positive_stride_16": avg_positive_diagnostics["positive_stride_16"],
        "positive_stride_32": avg_positive_diagnostics["positive_stride_32"],
        "positive_small": avg_positive_diagnostics["positive_small"],
        "positive_medium": avg_positive_diagnostics["positive_medium"],
        "positive_large": avg_positive_diagnostics["positive_large"],
        "learning_rate": last_lr,
    }


@torch.no_grad()
def validate_one_epoch(
    model,
    criterion,
    loader,
    device,
    epoch: int,
):
    model.eval()

    running_loss = 0.0
    running_cls_loss = 0.0
    running_obj_loss = 0.0
    running_box_loss = 0.0
    running_dfl_loss = 0.0
    running_positive_points = 0
    running_positive_diagnostics = {
        key: 0.0
        for key in POSITIVE_DIAGNOSTIC_KEYS
    }

    for images, targets, image_paths in loader:
        images = images.to(device, non_blocking=True)
        targets = [target.to(device, non_blocking=True) for target in targets]

        with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
            outputs = model(
                images,
                decode=False,
                return_aux=False,
            )

            loss_dict = criterion(outputs, targets)

        running_loss += loss_dict["loss"].item()
        running_cls_loss += loss_dict["cls_loss"].item()
        running_obj_loss += loss_dict["obj_loss"].item()
        running_box_loss += loss_dict["box_loss"].item()
        running_dfl_loss += loss_dict["dfl_loss"].item()
        running_positive_points += int(loss_dict["num_positive_points"].item())

        add_positive_diagnostics_from_loss(
            running=running_positive_diagnostics,
            loss_dict=loss_dict,
        )

    avg_loss = running_loss / len(loader)
    avg_cls_loss = running_cls_loss / len(loader)
    avg_obj_loss = running_obj_loss / len(loader)
    avg_box_loss = running_box_loss / len(loader)
    avg_dfl_loss = running_dfl_loss / len(loader)
    avg_pos = running_positive_points / len(loader)
    avg_diag = average_positive_diagnostics(
        running=running_positive_diagnostics,
        num_batches=len(loader),
    )

    print()
    print(
        f"Validation epoch {epoch} | "
        f"loss {avg_loss:.4f} | "
        f"cls {avg_cls_loss:.4f} | "
        f"obj {avg_obj_loss:.4f} | "
        f"box {avg_box_loss:.4f} | "
        f"dfl {avg_dfl_loss:.4f} | "
        f"pos {avg_pos:.1f} | "
        f"s8/s16/s32 "
        f"{avg_diag['positive_stride_8']:.1f}/"
        f"{avg_diag['positive_stride_16']:.1f}/"
        f"{avg_diag['positive_stride_32']:.1f} | "
        f"sm/md/lg "
        f"{avg_diag['positive_small']:.1f}/"
        f"{avg_diag['positive_medium']:.1f}/"
        f"{avg_diag['positive_large']:.1f}"
    )
    print()

    return {
        "loss": avg_loss,
        "cls_loss": avg_cls_loss,
        "obj_loss": avg_obj_loss,
        "box_loss": avg_box_loss,
        "dfl_loss": avg_dfl_loss,
        "positive_points": avg_pos,
        "positive_stride_8": avg_diag["positive_stride_8"],
        "positive_stride_16": avg_diag["positive_stride_16"],
        "positive_stride_32": avg_diag["positive_stride_32"],
        "positive_small": avg_diag["positive_small"],
        "positive_medium": avg_diag["positive_medium"],
        "positive_large": avg_diag["positive_large"],
    }


def main():
    project_root = Path(__file__).resolve().parents[1]
    dataset_root = project_root / "data" / "processed" / "voc2007_2012_custom_voc8"
    checkpoint_dir = project_root / "runs" / "checkpoints_voc8_v08_768_mobilenetv3_distilled"
    history_path = project_root / "runs" / "training_history_voc8_v08_768_mobilenetv3_distilled.csv"
    teacher_checkpoint_path = (
        project_root
        / "runs"
        / "checkpoints_voc8_v06_768_no_p2_regmax16_depth_slim"
        / "best.pt"
    )

    image_size = 768
    reg_max = 16
    num_classes = len(VOC_CLASSES)

    batch_size = 4
    num_epochs = 20
    warmup_epochs = 3
    close_mosaic_epochs = 5

    aux_loss_weight = 0.15

    learning_rate = 4e-4
    min_lr = learning_rate * 0.05
    weight_decay = 5e-4
    num_workers = 0

    train_split = "train"
    val_split = "val"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Project root: {project_root}")
    print(f"Dataset root: {dataset_root}")
    print(f"Device: {device}")
    print()
    print("Dataset split plan:")
    print("  train = VOC2007 train + VOC2007 val + VOC2012 train")
    print("  val   = VOC2007 test")
    print("  test  = VOC2012 val, used later by eval.py")
    print()

    train_dataset = VOCDatasetYOLO(
        root_dir=dataset_root,
        split=train_split,
        image_size=image_size,
        augment=True,
        mosaic_prob=0.50,
    )

    val_dataset = VOCDatasetYOLO(
        root_dir=dataset_root,
        split=val_split,
        image_size=image_size,
        augment=False,
        mosaic_prob=0.0,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=detection_collate_fn,
        num_workers=num_workers,
        pin_memory=True if device.type == "cuda" else False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=detection_collate_fn,
        num_workers=num_workers,
        pin_memory=True if device.type == "cuda" else False,
    )

    model = TinyYOLOAnchorFree(
        num_classes=num_classes,
        image_size=image_size,
        reg_max=reg_max,
        pretrained_backbone=True,
        use_auxiliary_heads=True,
    ).to(device)

    model_parameter_count = count_parameters(model)
    inference_parameter_count = count_inference_parameters(model)

    distiller = DetectionDistillationLoss(
        student_channels=(56, 72, 112),
        teacher_channels=(64, 80, 128),
        num_classes=num_classes,
        reg_max=reg_max,
        temperature=2.0,
        teacher_conf_threshold=0.05,
        feature_weight=0.20,
        classification_weight=0.40,
        objectness_weight=0.20,
        dfl_weight=0.30,
    ).to(device)
    distiller_parameter_count = count_parameters(distiller)

    if not teacher_checkpoint_path.exists():
        raise FileNotFoundError(
            "v0.6 teacher checkpoint not found. Distilled v0.8 training requires:\n"
            f"{teacher_checkpoint_path}"
        )

    teacher = YOLOv06Teacher(
        num_classes=num_classes,
        image_size=image_size,
        reg_max=reg_max,
    )

    teacher_checkpoint = torch.load(
        teacher_checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    load_v06_teacher_checkpoint(teacher, teacher_checkpoint)
    teacher_parameter_count = sum(p.numel() for p in teacher.parameters())
    del teacher_checkpoint
    teacher = teacher.to(device)

    ema = ModelEMA(model, decay=0.9998)

    criterion = YOLOAnchorFreeLoss(
        num_classes=num_classes,
        image_size=image_size,
        reg_max=reg_max,
        box_loss_weight=5.0,
        cls_loss_weight=1.0,
        obj_loss_weight=1.25,
        dfl_loss_weight=1.5,
        topk=10,
        alpha=0.5,
        beta=6.0,
        center_radius=2.5,
        focal_gamma=2.0,
        focal_alpha=0.25,
        objectness_quality_power=1.5,
        min_quality_target=0.05,
    ).to(device)

    # v0.8 architecture guard.
    assert model.image_size == image_size
    assert model.reg_max == reg_max
    assert model.num_bins == reg_max + 1
    assert model.strides == [8, 16, 32]

    expected_box_channels = 4 * (reg_max + 1)
    expected_head_channels = {
        "head_p3": 56,
        "head_p4": 72,
        "head_p5": 112,
        "aux_head_p3": 56,
        "aux_head_p4": 72,
        "aux_head_p5": 112,
    }

    for head_name, expected_input_channels in expected_head_channels.items():
        head = getattr(model, head_name)
        assert head is not None

        first_box_conv = head.box_branch[0].block[0]
        final_box_conv = head.box_branch[-1]

        assert first_box_conv.in_channels == expected_input_channels, (
            f"{head_name} has {first_box_conv.in_channels} input channels; "
            f"expected {expected_input_channels}."
        )
        assert first_box_conv.out_channels == max(48, expected_input_channels)
        assert final_box_conv.out_channels == expected_box_channels

    assert model.neck.p5_reduce.block[0].in_channels == 960
    assert model.neck.p5_reduce.block[0].out_channels == 112
    assert model.gold_fusion.n3_to_fusion.block[0].out_channels == 48
    assert len(model.neck.fpn3.blocks) == 3
    assert len(model.neck.fpn4.blocks) == 2
    assert len(model.neck.pan4.blocks) == 2
    assert len(model.neck.pan5.blocks) == 2
    assert len(model.gold_fusion.fuse[1].blocks) == 2
    assert not isinstance(model.gold_fusion.refine_n3, torch.nn.Identity)
    assert isinstance(model.gold_fusion.refine_n4, torch.nn.Identity)
    assert isinstance(model.gold_fusion.refine_n5, torch.nn.Identity)
    assert len(model.head_p3.extra_refine.blocks) == 3
    assert len(model.head_p4.extra_refine.blocks) == 2
    assert len(model.head_p5.extra_refine.blocks) == 2

    assert distiller.feature_adapters["n3"].in_channels == 56
    assert distiller.feature_adapters["n3"].out_channels == 64
    assert distiller.feature_adapters["n4"].in_channels == 72
    assert distiller.feature_adapters["n4"].out_channels == 80
    assert distiller.feature_adapters["n5"].in_channels == 112
    assert distiller.feature_adapters["n5"].out_channels == 128

    print("v0.8 architecture and distillation checks passed")

    optimizer = AdamW(
        list(model.parameters()) + list(distiller.parameters()),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    print(f"Train split:        {train_split}")
    print(f"Val split:          {val_split}")
    print(f"Image size:         {image_size}")
    print(f"Reg max:            {reg_max}")
    print("Backbone:           pretrained MobileNetV3-Large feature extractor")
    print("Neck:               width-slimmed 56/72/112 FPN/PAN")
    print("Fusion:             width 48; N3 refine kept, N4/N5 extra refine removed")
    print("Heads:              strides 8, 16, 32; hidden minimum 48")
    print("Prediction style:   anchor-free + quality-aware objectness + CIoU-style box loss + DFL")
    print("Architecture:       v0.8 compact MobileNetV3-Large distilled student")
    print("Box output:         4 x 17 = 68 DFL channels per scale")
    print("Auxiliary branch:   training-only aux heads")
    print(f"Aux loss weight:    {aux_loss_weight}")
    print("Distillation:       v0.6 frozen EfficientNet-B0 teacher")
    print(f"Teacher checkpoint: {teacher_checkpoint_path}")
    print("KD schedule:        epochs 1-2 off, 3-17 full, 18-20 half")
    print("KD weights:         feature 0.20, class 0.40, objectness 0.20, DFL 0.30")
    print("KD temperature:     2.0")
    print(f"Classes:            {VOC_CLASSES}")
    print(f"Num classes:        {num_classes}")
    print(f"Train images:       {len(train_dataset)}")
    print(f"Val images:         {len(val_dataset)}")
    print(f"Batch size:         {batch_size}")
    print(f"Epochs:             {num_epochs}")
    print(f"Warmup epochs:      {warmup_epochs}")
    print(f"Close mosaic last:  {close_mosaic_epochs} epochs")
    print(f"LR:                 {learning_rate}")
    print(f"Min LR:             {min_lr}")
    print(f"Weight decay:       {weight_decay}")
    print(f"Student train params:   {model_parameter_count:,}")
    print(f"Student infer params:   {inference_parameter_count:,}")
    print(f"KD adapter params:      {distiller_parameter_count:,}")
    print(f"Teacher infer params:   {teacher_parameter_count:,}")
    print(f"History:            {history_path}")
    print(f"Checkpoint folder:  {checkpoint_dir}")
    print(f"Experiment name:    {EXPERIMENT_NAME}")
    print(f"Experiment:         {EXPERIMENT_DESCRIPTION}")
    print()

    latest_path = checkpoint_dir / "latest.pt"
    best_path = checkpoint_dir / "best.pt"
    interrupt_path = checkpoint_dir / "interrupt.pt"

    # Resume behavior:
    # resume_training=True means the script will continue from latest.pt if it exists.
    # start_fresh=True means the script will delete old checkpoints/history and restart.
    # Keep start_fresh=False if you want to shut down your laptop and continue tomorrow.
    resume_training = True
    start_fresh = False
    resume_checkpoint_path = latest_path

    print("Resume-safe current training run for the VOC2007 + VOC2012 custom split.")
    print(r"This uses one stable checkpoint folder: runs\checkpoints_voc8_v08_768_mobilenetv3_distilled")
    print("A checkpoint resumes only if it belongs to this exact experiment name.")
    print("Different/older experiment checkpoints are ignored and removed from the stable folder.")
    print("Current split:")
    print("  train = VOC2007 train + VOC2007 val + VOC2012 train")
    print("  val   = VOC2007 test")
    print()
    print("Resume settings:")
    print(f"  resume_training: {resume_training}")
    print(f"  start_fresh:     {start_fresh}")
    print(f"  resume path:     {resume_checkpoint_path}")
    print(f"  latest path:     {latest_path}")
    print(f"  best path:       {best_path}")
    print(f"  interrupt path:  {interrupt_path}")
    print()

    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    if start_fresh:
        for old_checkpoint in [latest_path, best_path, interrupt_path]:
            if old_checkpoint.exists():
                old_checkpoint.unlink()
                print(f"Removed old checkpoint: {old_checkpoint}")

        if history_path.exists():
            history_path.unlink()
            print(f"Removed old history file: {history_path}")

        print()

    start_epoch = 1
    best_val_loss = float("inf")

    if resume_training and not start_fresh and resume_checkpoint_path.exists():
        can_resume, reason, _ = inspect_checkpoint_for_resume(
            checkpoint_path=resume_checkpoint_path,
            expected_experiment_name=EXPERIMENT_NAME,
            expected_parameter_count=model_parameter_count,
        )

        if can_resume:
            checkpoint, start_epoch, best_val_loss = load_training_checkpoint(
                checkpoint_path=resume_checkpoint_path,
                model=model,
                ema=ema,
                distiller=distiller,
                optimizer=optimizer,
                scaler=scaler,
                device=device,
                expected_experiment_name=EXPERIMENT_NAME,
                expected_parameter_count=model_parameter_count,
            )

            print("Resumed training checkpoint from this same experiment.")
            print(f"  checkpoint:       {resume_checkpoint_path}")
            print(f"  experiment:       {checkpoint.get('experiment_name')}")
            print(f"  completed epoch:  {checkpoint.get('epoch')}")
            print(f"  next epoch:       {start_epoch}")
            print(f"  checkpoint val:   {float(checkpoint.get('val_loss', float('nan'))):.4f}")
            print(f"  best val loss:    {best_val_loss:.4f}")
            print(f"  EMA updates:      {ema.updates}")
            print()
        else:
            remove_stale_training_files(
                paths=[latest_path, best_path, interrupt_path, history_path],
                reason=reason,
            )
    elif resume_training and not start_fresh:
        print("No latest checkpoint found, so this version will start from epoch 1.")
        print("After it saves latest.pt, future runs will resume from that checkpoint.")
        print()
    else:
        print("Resume disabled or fresh start requested, so training will start from epoch 1.")
        print()

    if start_epoch > num_epochs:
        print(
            f"Checkpoint already completed epoch {start_epoch - 1}, "
            f"which is >= configured num_epochs {num_epochs}."
        )
        print("Increase num_epochs if you want to continue training for more epochs.")
        return

    completed_epoch = start_epoch - 1

    try:
        for epoch in range(start_epoch, num_epochs + 1):
            if epoch > num_epochs - close_mosaic_epochs:
                train_dataset.mosaic_prob = 0.0
            else:
                train_dataset.mosaic_prob = 0.50

            print("=" * 80)
            print(f"Starting epoch {epoch}/{num_epochs}")
            print(f"Mosaic probability: {train_dataset.mosaic_prob}")
            print("=" * 80)

            train_metrics = train_one_epoch(
                model=model,
                teacher=teacher,
                distiller=distiller,
                ema=ema,
                criterion=criterion,
                loader=train_loader,
                optimizer=optimizer,
                scaler=scaler,
                device=device,
                epoch=epoch,
                num_epochs=num_epochs,
                base_lr=learning_rate,
                min_lr=min_lr,
                warmup_epochs=warmup_epochs,
                aux_loss_weight=aux_loss_weight,
                log_interval=25,
            )

            val_metrics = validate_one_epoch(
                model=ema.ema,
                criterion=criterion,
                loader=val_loader,
                device=device,
                epoch=epoch,
            )

            if val_metrics["loss"] < best_val_loss:
                best_val_loss = val_metrics["loss"]
                is_best_epoch = True
            else:
                is_best_epoch = False

            save_checkpoint(
                checkpoint_path=latest_path,
                model=model,
                ema=ema,
                distiller=distiller,
                optimizer=optimizer,
                scaler=scaler,
                epoch=epoch,
                train_loss=train_metrics["loss"],
                val_loss=val_metrics["loss"],
                best_val_loss=best_val_loss,
                experiment_name=EXPERIMENT_NAME,
                experiment_description=EXPERIMENT_DESCRIPTION,
                model_parameter_count=model_parameter_count,
            )

            print(f"Saved latest checkpoint: {latest_path}")

            if is_best_epoch:
                save_checkpoint(
                    checkpoint_path=best_path,
                    model=model,
                    ema=ema,
                    distiller=distiller,
                    optimizer=optimizer,
                    scaler=scaler,
                    epoch=epoch,
                    train_loss=train_metrics["loss"],
                    val_loss=val_metrics["loss"],
                    best_val_loss=best_val_loss,
                    experiment_name=EXPERIMENT_NAME,
                    experiment_description=EXPERIMENT_DESCRIPTION,
                    model_parameter_count=model_parameter_count,
                )

                print(f"Saved best checkpoint:   {best_path}")

            append_history(
                history_path=history_path,
                epoch=epoch,
                learning_rate=train_metrics["learning_rate"],
                train_metrics=train_metrics,
                val_metrics=val_metrics,
                mosaic_prob=train_dataset.mosaic_prob,
            )

            completed_epoch = epoch

            print(
                f"Epoch {epoch} summary | "
                f"train total loss {train_metrics['loss']:.4f} | "
                f"main {train_metrics['main_loss']:.4f} | "
                f"aux {train_metrics['aux_loss']:.4f} | "
                f"kd {train_metrics['distillation_loss']:.4f} "
                f"x {train_metrics['distillation_scale']:.2f} | "
                f"val loss {val_metrics['loss']:.4f} | "
                f"best val loss {best_val_loss:.4f}"
            )
            print()

    except KeyboardInterrupt:
        print()
        print("Training interrupted by user.")
        print("Progress is safe up to the last completed epoch.")

        if completed_epoch >= start_epoch:
            print(f"Last completed epoch: {completed_epoch}")
            print(f"Resume later from: {latest_path}")
        else:
            print("No full epoch completed in this run before interruption.")
            print("Resume will use the previous latest.pt if it existed.")

        save_checkpoint(
            checkpoint_path=interrupt_path,
            model=model,
            ema=ema,
            distiller=distiller,
            optimizer=optimizer,
            scaler=scaler,
            epoch=completed_epoch,
            train_loss=float("nan"),
            val_loss=float("nan"),
            best_val_loss=best_val_loss,
            experiment_name=EXPERIMENT_NAME,
            experiment_description=EXPERIMENT_DESCRIPTION,
            model_parameter_count=model_parameter_count,
        )
        print(f"Saved interrupt snapshot: {interrupt_path}")
        print("Tomorrow, run python src\train.py again to continue from latest.pt.")
        return

    print("Training complete.")
    print(f"Best val loss: {best_val_loss:.4f}")
    print(f"Best checkpoint: {best_path}")
    print(f"History file: {history_path}")


if __name__ == "__main__":
    main()