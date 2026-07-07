from pathlib import Path
import random

import matplotlib.pyplot as plt
from PIL import Image
import matplotlib.patches as patches


VOC_CLASSES = [
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
]


def yolo_to_xyxy(
    x_center_norm: float,
    y_center_norm: float,
    width_norm: float,
    height_norm: float,
    image_width: int,
    image_height: int,
) -> tuple[float, float, float, float]:
    """
    Convert YOLO format back to corner format.

    YOLO:
        x_center_norm, y_center_norm, width_norm, height_norm

    Corner:
        x_min, y_min, x_max, y_max
    """
    x_center = x_center_norm * image_width
    y_center = y_center_norm * image_height
    box_width = width_norm * image_width
    box_height = height_norm * image_height

    x_min = x_center - box_width / 2
    y_min = y_center - box_height / 2
    x_max = x_center + box_width / 2
    y_max = y_center + box_height / 2

    return x_min, y_min, x_max, y_max


def load_labels(label_path: Path, image_width: int, image_height: int):
    boxes = []

    if not label_path.exists():
        return boxes

    text = label_path.read_text(encoding="utf-8").strip()

    if not text:
        return boxes

    for line in text.splitlines():
        parts = line.split()

        if len(parts) != 5:
            raise ValueError(f"Invalid label line in {label_path}: {line}")

        class_id = int(parts[0])
        x_center_norm = float(parts[1])
        y_center_norm = float(parts[2])
        width_norm = float(parts[3])
        height_norm = float(parts[4])

        x_min, y_min, x_max, y_max = yolo_to_xyxy(
            x_center_norm,
            y_center_norm,
            width_norm,
            height_norm,
            image_width,
            image_height,
        )

        boxes.append(
            {
                "class_id": class_id,
                "class_name": VOC_CLASSES[class_id],
                "x_min": x_min,
                "y_min": y_min,
                "x_max": x_max,
                "y_max": y_max,
            }
        )

    return boxes


def draw_image_with_boxes(image_path: Path, label_path: Path, output_path: Path) -> None:
    image = Image.open(image_path).convert("RGB")
    image_width, image_height = image.size

    boxes = load_labels(label_path, image_width, image_height)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(image)
    ax.axis("off")
    ax.set_title(f"{image_path.name} | {len(boxes)} objects")

    for box in boxes:
        x_min = box["x_min"]
        y_min = box["y_min"]
        width = box["x_max"] - box["x_min"]
        height = box["y_max"] - box["y_min"]

        rect = patches.Rectangle(
            (x_min, y_min),
            width,
            height,
            linewidth=2,
            edgecolor="red",
            facecolor="none",
        )
        ax.add_patch(rect)

        ax.text(
            x_min,
            max(0, y_min - 5),
            box["class_name"],
            fontsize=10,
            color="white",
            bbox={"facecolor": "red", "alpha": 0.7, "pad": 2},
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)


def visualize_random_samples(split: str = "train", num_samples: int = 8) -> None:
    project_root = Path(__file__).resolve().parents[1]

    image_dir = project_root / "data" / "processed" / "voc_yolo" / "images" / split
    label_dir = project_root / "data" / "processed" / "voc_yolo" / "labels" / split
    output_dir = project_root / "outputs" / "label_visualization" / split

    image_paths = sorted(image_dir.glob("*.jpg"))

    if not image_paths:
        raise FileNotFoundError(f"No images found in {image_dir}")

    random.seed(42)
    sample_paths = random.sample(image_paths, min(num_samples, len(image_paths)))

    print(f"Visualizing {len(sample_paths)} random samples from split: {split}")
    print(f"Output folder: {output_dir}")
    print()

    for image_path in sample_paths:
        label_path = label_dir / f"{image_path.stem}.txt"
        output_path = output_dir / f"{image_path.stem}_boxes.jpg"

        draw_image_with_boxes(image_path, label_path, output_path)

        print(f"Saved: {output_path}")

    print()
    print("Done.")


if __name__ == "__main__":
    visualize_random_samples(split="train", num_samples=8)