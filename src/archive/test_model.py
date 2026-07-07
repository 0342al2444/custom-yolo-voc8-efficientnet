from pathlib import Path

import torch
from torch.utils.data import DataLoader

from dataset import VOCDatasetYOLO, detection_collate_fn
from model import TinyYOLOAnchorFree, count_parameters


def print_raw_outputs(outputs):
    for i, output in enumerate(outputs):
        box_raw = output["box_raw"]
        cls_logits = output["cls_logits"]
        stride = output["stride"]

        print(f"Scale {i}: stride {stride}")
        print(f"  box_raw shape:   {box_raw.shape}")
        print(f"  cls_logits shape:{cls_logits.shape}")


def main():
    project_root = Path(__file__).resolve().parents[1]
    dataset_root = project_root / "data" / "processed" / "voc_yolo"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    dataset = VOCDatasetYOLO(
        root_dir=dataset_root,
        split="train",
        image_size=416,
    )

    loader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=True,
        collate_fn=detection_collate_fn,
    )

    images, targets, image_paths = next(iter(loader))

    print()
    print("Input batch")
    print(f"  images shape: {images.shape}")
    print(f"  number of target lists: {len(targets)}")
    print(f"  objects in image 0: {len(targets[0])}")
    print(f"  objects in image 1: {len(targets[1])}")

    model = TinyYOLOAnchorFree(
        num_classes=20,
        image_size=416,
    ).to(device)

    print()
    print(f"Trainable parameters: {count_parameters(model):,}")

    images = images.to(device)

    model.eval()

    with torch.no_grad():
        raw_outputs = model(images, decode=False)
        decoded_outputs = model(images, decode=True)

    print()
    print("Raw outputs")
    print_raw_outputs(raw_outputs)

    print()
    print("Decoded outputs")
    print(f"  boxes shape:  {decoded_outputs['boxes'].shape}")
    print(f"  scores shape: {decoded_outputs['scores'].shape}")

    print()
    print("Meaning:")
    print("  boxes shape [B, N, 4]")
    print("  scores shape [B, N, 20]")
    print("  B = batch size")
    print("  N = total feature points from 52x52 + 26x26 + 13x13")
    print("  52x52 + 26x26 + 13x13 = 3549")

    print()
    print("First 5 decoded boxes for image 0:")
    print(decoded_outputs["boxes"][0, :5].cpu())

    print()
    print("First 5 class score vectors for image 0:")
    print(decoded_outputs["scores"][0, :5].cpu())

    print()
    print("Done.")


if __name__ == "__main__":
    main()