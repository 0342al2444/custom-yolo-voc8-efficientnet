from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader

from dataset import VOCDatasetYOLO, detection_collate_fn, VOC_CLASSES


def draw_targets_on_image(image_tensor, targets, output_path: Path):
    """
    Save one image with boxes after the Dataset preprocessing.

    This checks that letterbox resizing did not break the labels.
    """
    image_np = image_tensor.permute(1, 2, 0).numpy()
    image_np = (image_np * 255).clip(0, 255).astype(np.uint8)

    image = Image.fromarray(image_np)
    draw = ImageDraw.Draw(image)

    image_width, image_height = image.size

    for target in targets:
        class_id = int(target[0].item())
        x_center = target[1].item() * image_width
        y_center = target[2].item() * image_height
        box_width = target[3].item() * image_width
        box_height = target[4].item() * image_height

        x_min = x_center - box_width / 2
        y_min = y_center - box_height / 2
        x_max = x_center + box_width / 2
        y_max = y_center + box_height / 2

        class_name = VOC_CLASSES[class_id]

        draw.rectangle(
            [x_min, y_min, x_max, y_max],
            outline="red",
            width=3,
        )

        text_position = (x_min, max(0, y_min - 14))
        draw.rectangle(
            [
                text_position[0],
                text_position[1],
                text_position[0] + 8 * len(class_name),
                text_position[1] + 14,
            ],
            fill="red",
        )
        draw.text(text_position, class_name, fill="white")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def main():
    project_root = Path(__file__).resolve().parents[1]

    dataset_root = project_root / "data" / "processed" / "voc_yolo"
    output_dir = project_root / "outputs" / "dataset_check"

    dataset = VOCDatasetYOLO(
        root_dir=dataset_root,
        split="train",
        image_size=416,
    )

    print(f"Dataset length: {len(dataset)}")

    image, targets, image_path = dataset[0]

    print()
    print("One sample:")
    print(f"Image path: {image_path}")
    print(f"Image tensor shape: {image.shape}")
    print(f"Image tensor min: {image.min().item():.4f}")
    print(f"Image tensor max: {image.max().item():.4f}")
    print(f"Targets shape: {targets.shape}")
    print("First targets:")
    print(targets[:5])

    loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        collate_fn=detection_collate_fn,
    )

    images, batch_targets, image_paths = next(iter(loader))

    print()
    print("One batch:")
    print(f"Batch image tensor shape: {images.shape}")
    print(f"Number of target lists: {len(batch_targets)}")

    for i in range(len(batch_targets)):
        print(f"Image {i}: {len(batch_targets[i])} objects")

    for i in range(min(4, len(batch_targets))):
        output_path = output_dir / f"sample_{i}_letterbox.jpg"
        draw_targets_on_image(images[i], batch_targets[i], output_path)
        print(f"Saved visual check: {output_path}")

    print()
    print("Done.")


if __name__ == "__main__":
    main()