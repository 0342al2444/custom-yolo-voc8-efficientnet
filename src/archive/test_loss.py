from pathlib import Path

import torch
from torch.utils.data import DataLoader

from dataset import VOCDatasetYOLO, detection_collate_fn
from model import TinyYOLOAnchorFree
from loss import YOLOAnchorFreeLoss


def main():
    project_root = Path(__file__).resolve().parents[1]
    dataset_root = project_root / "data" / "processed" / "voc_yolo"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    dataset = VOCDatasetYOLO(
        root_dir=dataset_root,
        split="train",
        image_size=416,
        augment=True,
    )

    loader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=True,
        collate_fn=detection_collate_fn,
    )

    images, targets, image_paths = next(iter(loader))

    images = images.to(device)
    targets = [target.to(device) for target in targets]

    model = TinyYOLOAnchorFree(
        num_classes=20,
        image_size=416,
        reg_max=16,
    ).to(device)

    criterion = YOLOAnchorFreeLoss(
        num_classes=20,
        image_size=416,
        reg_max=16,
        box_loss_weight=5.0,
        cls_loss_weight=1.0,
        dfl_loss_weight=1.5,
        max_points_per_object=9,
        center_radius=2.5,
    ).to(device)

    model.train()

    outputs = model(images, decode=False)

    loss_dict = criterion(outputs, targets)

    loss = loss_dict["loss"]

    print()
    print("Loss sanity check")
    print(f"Total loss:          {loss.item():.4f}")
    print(f"Class loss:          {loss_dict['cls_loss'].item():.4f}")
    print(f"Box loss:            {loss_dict['box_loss'].item():.4f}")
    print(f"DFL loss:            {loss_dict['dfl_loss'].item():.4f}")
    print(f"Positive points:     {int(loss_dict['num_positive_points'].item())}")

    print()
    print("Testing backward pass...")

    loss.backward()

    first_param = next(model.parameters())

    if first_param.grad is None:
        print("Gradient check: FAILED, no gradient found.")
    else:
        grad_mean = first_param.grad.abs().mean().item()
        print(f"Gradient check: OK, mean abs grad = {grad_mean:.8f}")

    print()
    print("Done.")


if __name__ == "__main__":
    main()