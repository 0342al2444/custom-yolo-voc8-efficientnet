from pathlib import Path
import random

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from dataset import VOC_CLASSES
from model import TinyYOLOAnchorFree


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
):
    all_boxes = []
    all_scores = []
    all_classes = []

    num_classes = scores.shape[1]

    for class_id in range(num_classes):
        class_scores = scores[:, class_id]
        keep = class_scores >= confidence_threshold

        if keep.sum() == 0:
            continue

        class_boxes = boxes[keep]
        class_scores = class_scores[keep]

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
            torch.zeros((0, 4), device=boxes.device),
            torch.zeros((0,), device=boxes.device),
            torch.zeros((0,), dtype=torch.long, device=boxes.device),
        )

    final_boxes = torch.cat(all_boxes, dim=0)
    final_scores = torch.cat(all_scores, dim=0)
    final_classes = torch.cat(all_classes, dim=0)

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


@torch.no_grad()
def predict_one_image(
    model,
    image_path: Path,
    output_path: Path,
    device,
    image_size: int,
    confidence_threshold: float,
    nms_iou_threshold: float,
    max_detections: int,
):
    image = Image.open(image_path).convert("RGB")
    original_width, original_height = image.size

    tensor, scale, pad_x, pad_y = letterbox_image(
        image=image,
        image_size=image_size,
    )

    tensor = tensor.to(device)

    decoded = model(
        tensor,
        decode=True,
    )

    boxes = decoded["boxes"][0]
    scores = decoded["scores"][0]

    final_boxes, final_scores, final_classes = collect_predictions(
        boxes=boxes,
        scores=scores,
        confidence_threshold=confidence_threshold,
        nms_iou_threshold=nms_iou_threshold,
        max_detections=max_detections,
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
    project_root = Path(__file__).resolve().parents[1]

    dataset_root = project_root / "data" / "processed" / "voc2007_2012_custom_voc8"
    checkpoint_path = project_root / "runs" / "checkpoints_voc2007_2012_custom_voc8" / "best.pt"
    output_dir = project_root / "runs" / "predictions_voc2007_2012_custom_voc8"

    # "val"  = VOC2007 test
    # "test" = VOC2012 val
    split = "test"

    image_size = 640
    num_classes = len(VOC_CLASSES)

    confidence_threshold = 0.20
    nms_iou_threshold = 0.50
    max_detections = 50
    num_images = 12

    # None means different random images every run.
    # Use 42 if you want the same random images every run for fair visual comparison.
    random_seed = None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    image_dir, image_paths = find_image_paths(
        dataset_root=dataset_root,
        split=split,
    )

    print(f"Device: {device}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Dataset root: {dataset_root}")
    print(f"Dataset split: {split}")
    print('Split meaning: "val" = VOC2007 test, "test" = VOC2012 val')
    print(f"Detected image directory: {image_dir}")
    print(f"Images found: {len(image_paths)}")
    print(f"Output directory: {output_dir}")
    print(f"Classes: {VOC_CLASSES}")
    print()

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}. "
            "Train first or check the checkpoint path."
        )

    if not image_paths:
        raise FileNotFoundError(
            "No images found.\n"
            f"Tried:\n"
            f"  {dataset_root / split / 'images'}\n"
            f"  {dataset_root / 'images' / split}\n"
            f"  {dataset_root / split}\n"
        )

    if random_seed is not None:
        random.seed(random_seed)

    selected_paths = random.sample(
        image_paths,
        k=min(num_images, len(image_paths)),
    )

    model = TinyYOLOAnchorFree(
        num_classes=num_classes,
        image_size=image_size,
        reg_max=16,
        pretrained_backbone=False,
        use_auxiliary_heads=True,
    ).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print(f"Loaded epoch: {checkpoint['epoch']}")
    print(f"Checkpoint val loss: {checkpoint['val_loss']:.4f}")
    print(f"Confidence threshold: {confidence_threshold}")
    print(f"NMS IoU threshold: {nms_iou_threshold}")
    print(f"Max detections/image: {max_detections}")
    print(f"Random images: True")
    print(f"Random seed: {random_seed}")
    print()

    for idx, image_path in enumerate(selected_paths):
        output_path = output_dir / f"prediction_{idx}.jpg"

        num_predictions = predict_one_image(
            model=model,
            image_path=image_path,
            output_path=output_path,
            device=device,
            image_size=image_size,
            confidence_threshold=confidence_threshold,
            nms_iou_threshold=nms_iou_threshold,
            max_detections=max_detections,
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