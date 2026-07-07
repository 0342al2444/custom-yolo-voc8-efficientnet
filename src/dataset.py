from pathlib import Path
from typing import List, Tuple
import random

import numpy as np
from PIL import Image, ImageEnhance
import torch
from torch.utils.data import Dataset


ORIGINAL_VOC_CLASSES = [
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


SELECTED_CLASSES = [
    "person",
    "car",
    "dog",
    "cat",
    "bus",
    "train",
    "bicycle",
    "aeroplane",
]

VOC_CLASSES = SELECTED_CLASSES

OLD_CLASS_TO_NEW_CLASS = {
    ORIGINAL_VOC_CLASSES.index(class_name): new_id
    for new_id, class_name in enumerate(SELECTED_CLASSES)
}


def xywh_norm_to_xyxy_pixels(
    labels: np.ndarray,
    image_width: int,
    image_height: int,
) -> np.ndarray:
    if labels.size == 0:
        return np.zeros((0, 5), dtype=np.float32)

    output = labels.copy().astype(np.float32)

    class_ids = output[:, 0]
    x_center = output[:, 1] * image_width
    y_center = output[:, 2] * image_height
    box_width = output[:, 3] * image_width
    box_height = output[:, 4] * image_height

    x_min = x_center - box_width / 2.0
    y_min = y_center - box_height / 2.0
    x_max = x_center + box_width / 2.0
    y_max = y_center + box_height / 2.0

    output[:, 0] = class_ids
    output[:, 1] = x_min
    output[:, 2] = y_min
    output[:, 3] = x_max
    output[:, 4] = y_max

    return output


def xyxy_pixels_to_xywh_norm(
    labels: np.ndarray,
    image_size: int,
) -> np.ndarray:
    if labels.size == 0:
        return np.zeros((0, 5), dtype=np.float32)

    output = labels.copy().astype(np.float32)

    class_ids = output[:, 0]
    x_min = output[:, 1]
    y_min = output[:, 2]
    x_max = output[:, 3]
    y_max = output[:, 4]

    x_center = (x_min + x_max) / 2.0
    y_center = (y_min + y_max) / 2.0
    box_width = x_max - x_min
    box_height = y_max - y_min

    output[:, 0] = class_ids
    output[:, 1] = x_center / image_size
    output[:, 2] = y_center / image_size
    output[:, 3] = box_width / image_size
    output[:, 4] = box_height / image_size

    return output


def xyxy_pixels_to_xywh_norm_rect(
    labels: np.ndarray,
    image_width: int,
    image_height: int,
) -> np.ndarray:
    if labels.size == 0:
        return np.zeros((0, 5), dtype=np.float32)

    class_ids = labels[:, 0]
    x_min = labels[:, 1]
    y_min = labels[:, 2]
    x_max = labels[:, 3]
    y_max = labels[:, 4]

    x_center = (x_min + x_max) / 2.0 / image_width
    y_center = (y_min + y_max) / 2.0 / image_height
    box_width = (x_max - x_min) / image_width
    box_height = (y_max - y_min) / image_height

    output = np.stack(
        [class_ids, x_center, y_center, box_width, box_height],
        axis=1,
    ).astype(np.float32)

    return output


def drop_invalid_boxes(labels_xyxy: np.ndarray, min_size: float = 4.0) -> np.ndarray:
    if labels_xyxy.size == 0:
        return labels_xyxy

    widths = labels_xyxy[:, 3] - labels_xyxy[:, 1]
    heights = labels_xyxy[:, 4] - labels_xyxy[:, 2]

    keep = (widths >= min_size) & (heights >= min_size)

    return labels_xyxy[keep]


def random_horizontal_flip(
    image: Image.Image,
    labels: np.ndarray,
    p: float = 0.5,
) -> tuple[Image.Image, np.ndarray]:
    if random.random() > p:
        return image, labels

    image = image.transpose(Image.FLIP_LEFT_RIGHT)

    if labels.size > 0:
        labels = labels.copy()
        labels[:, 1] = 1.0 - labels[:, 1]

    return image, labels


def random_color_jitter(
    image: Image.Image,
    p: float = 0.8,
) -> Image.Image:
    if random.random() > p:
        return image

    brightness = random.uniform(0.75, 1.25)
    contrast = random.uniform(0.75, 1.25)
    saturation = random.uniform(0.75, 1.25)

    image = ImageEnhance.Brightness(image).enhance(brightness)
    image = ImageEnhance.Contrast(image).enhance(contrast)
    image = ImageEnhance.Color(image).enhance(saturation)

    return image


def random_scale_translate(
    image: Image.Image,
    labels: np.ndarray,
    scale_range: tuple[float, float] = (0.75, 1.25),
    translate_frac: float = 0.12,
    p: float = 0.8,
    fill_color: tuple[int, int, int] = (114, 114, 114),
) -> tuple[Image.Image, np.ndarray]:
    if random.random() > p:
        return image, labels

    image_width, image_height = image.size

    labels_xyxy = xywh_norm_to_xyxy_pixels(
        labels,
        image_width,
        image_height,
    )

    scale = random.uniform(scale_range[0], scale_range[1])

    max_dx = translate_frac * image_width
    max_dy = translate_frac * image_height

    dx = random.uniform(-max_dx, max_dx)
    dy = random.uniform(-max_dy, max_dy)

    inverse_scale = 1.0 / scale

    affine_matrix = (
        inverse_scale,
        0.0,
        -dx * inverse_scale,
        0.0,
        inverse_scale,
        -dy * inverse_scale,
    )

    image = image.transform(
        size=(image_width, image_height),
        method=Image.AFFINE,
        data=affine_matrix,
        resample=Image.BILINEAR,
        fillcolor=fill_color,
    )

    if labels_xyxy.size > 0:
        labels_xyxy[:, 1] = labels_xyxy[:, 1] * scale + dx
        labels_xyxy[:, 2] = labels_xyxy[:, 2] * scale + dy
        labels_xyxy[:, 3] = labels_xyxy[:, 3] * scale + dx
        labels_xyxy[:, 4] = labels_xyxy[:, 4] * scale + dy

        labels_xyxy[:, 1] = np.clip(labels_xyxy[:, 1], 0, image_width)
        labels_xyxy[:, 2] = np.clip(labels_xyxy[:, 2], 0, image_height)
        labels_xyxy[:, 3] = np.clip(labels_xyxy[:, 3], 0, image_width)
        labels_xyxy[:, 4] = np.clip(labels_xyxy[:, 4], 0, image_height)

        labels_xyxy = drop_invalid_boxes(labels_xyxy, min_size=4.0)

    labels = xyxy_pixels_to_xywh_norm_rect(
        labels_xyxy,
        image_width=image_width,
        image_height=image_height,
    )

    return image, labels


def apply_train_augmentations(
    image: Image.Image,
    labels: np.ndarray,
) -> tuple[Image.Image, np.ndarray]:
    image, labels = random_horizontal_flip(image, labels, p=0.5)
    image, labels = random_scale_translate(image, labels, p=0.8)
    image = random_color_jitter(image, p=0.8)

    return image, labels


def letterbox_image_and_labels(
    image: Image.Image,
    labels: np.ndarray,
    image_size: int,
    fill_color: Tuple[int, int, int] = (114, 114, 114),
) -> Tuple[Image.Image, np.ndarray]:
    original_width, original_height = image.size

    scale = min(image_size / original_width, image_size / original_height)

    new_width = int(round(original_width * scale))
    new_height = int(round(original_height * scale))

    resized_image = image.resize((new_width, new_height), Image.BILINEAR)

    pad_left = (image_size - new_width) // 2
    pad_top = (image_size - new_height) // 2

    canvas = Image.new("RGB", (image_size, image_size), fill_color)
    canvas.paste(resized_image, (pad_left, pad_top))

    labels_xyxy = xywh_norm_to_xyxy_pixels(
        labels,
        original_width,
        original_height,
    )

    if labels_xyxy.size > 0:
        labels_xyxy[:, 1] = labels_xyxy[:, 1] * scale + pad_left
        labels_xyxy[:, 2] = labels_xyxy[:, 2] * scale + pad_top
        labels_xyxy[:, 3] = labels_xyxy[:, 3] * scale + pad_left
        labels_xyxy[:, 4] = labels_xyxy[:, 4] * scale + pad_top

        labels_xyxy[:, 1] = np.clip(labels_xyxy[:, 1], 0, image_size)
        labels_xyxy[:, 2] = np.clip(labels_xyxy[:, 2], 0, image_size)
        labels_xyxy[:, 3] = np.clip(labels_xyxy[:, 3], 0, image_size)
        labels_xyxy[:, 4] = np.clip(labels_xyxy[:, 4], 0, image_size)

        labels_xyxy = drop_invalid_boxes(labels_xyxy, min_size=4.0)

    labels_yolo = xyxy_pixels_to_xywh_norm(labels_xyxy, image_size)

    return canvas, labels_yolo


class VOCDatasetYOLO(Dataset):
    def __init__(
        self,
        root_dir: str | Path,
        split: str = "train",
        image_size: int = 640,
        augment: bool | None = None,
        mosaic_prob: float = 0.50,
    ):
        self.root_dir = Path(root_dir)
        self.split = split
        self.image_size = image_size
        self.mosaic_prob = mosaic_prob

        if augment is None:
            augment = split in {"train", "trainval"}

        self.augment = augment

        if split == "trainval":
            split_names = ["train", "val"]
        else:
            split_names = [split]

        self.samples = []

        for split_name in split_names:
            image_dir = self.root_dir / "images" / split_name
            label_dir = self.root_dir / "labels" / split_name

            if not image_dir.exists():
                raise FileNotFoundError(f"Image folder not found: {image_dir}")

            if not label_dir.exists():
                raise FileNotFoundError(f"Label folder not found: {label_dir}")

            image_paths = sorted(image_dir.glob("*.jpg"))

            for image_path in image_paths:
                label_path = label_dir / f"{image_path.stem}.txt"
                self.samples.append((image_path, label_path))

        if not self.samples:
            raise FileNotFoundError(f"No JPG images found for split: {split}")

    def __len__(self) -> int:
        return len(self.samples)

    def load_labels(self, label_path: Path) -> np.ndarray:
        if not label_path.exists():
            return np.zeros((0, 5), dtype=np.float32)

        text = label_path.read_text(encoding="utf-8").strip()

        if not text:
            return np.zeros((0, 5), dtype=np.float32)

        labels: List[List[float]] = []

        for line in text.splitlines():
            parts = line.strip().split()

            if len(parts) != 5:
                raise ValueError(f"Invalid label line in {label_path}: {line}")

            old_class_id = int(parts[0])

            if old_class_id not in OLD_CLASS_TO_NEW_CLASS:
                continue

            new_class_id = OLD_CLASS_TO_NEW_CLASS[old_class_id]

            x_center = float(parts[1])
            y_center = float(parts[2])
            width = float(parts[3])
            height = float(parts[4])

            labels.append([new_class_id, x_center, y_center, width, height])

        if not labels:
            return np.zeros((0, 5), dtype=np.float32)

        return np.array(labels, dtype=np.float32)

    def load_raw_sample(self, index: int):
        image_path, label_path = self.samples[index]

        image = Image.open(image_path).convert("RGB")
        labels = self.load_labels(label_path)

        return image, labels, image_path

    def load_mosaic(self, index: int):
        mosaic_size = self.image_size
        half_size = mosaic_size // 2

        indices = [index]

        while len(indices) < 4:
            indices.append(random.randint(0, len(self.samples) - 1))

        random.shuffle(indices)

        canvas = Image.new("RGB", (mosaic_size, mosaic_size), (114, 114, 114))

        offsets = [
            (0, 0),
            (half_size, 0),
            (0, half_size),
            (half_size, half_size),
        ]

        all_labels = []
        first_image_path = None

        for sample_index, (offset_x, offset_y) in zip(indices, offsets):
            image, labels, image_path = self.load_raw_sample(sample_index)

            if first_image_path is None:
                first_image_path = image_path

            image, labels = random_horizontal_flip(image, labels, p=0.5)
            image = random_color_jitter(image, p=0.8)

            image, labels = letterbox_image_and_labels(
                image=image,
                labels=labels,
                image_size=half_size,
            )

            canvas.paste(image, (offset_x, offset_y))

            labels_xyxy = xywh_norm_to_xyxy_pixels(
                labels,
                half_size,
                half_size,
            )

            if labels_xyxy.size > 0:
                labels_xyxy[:, 1] += offset_x
                labels_xyxy[:, 2] += offset_y
                labels_xyxy[:, 3] += offset_x
                labels_xyxy[:, 4] += offset_y

                labels_xyxy[:, 1] = np.clip(labels_xyxy[:, 1], 0, mosaic_size)
                labels_xyxy[:, 2] = np.clip(labels_xyxy[:, 2], 0, mosaic_size)
                labels_xyxy[:, 3] = np.clip(labels_xyxy[:, 3], 0, mosaic_size)
                labels_xyxy[:, 4] = np.clip(labels_xyxy[:, 4], 0, mosaic_size)

                labels_xyxy = drop_invalid_boxes(labels_xyxy, min_size=4.0)

                all_labels.append(labels_xyxy)

        if all_labels:
            labels_xyxy = np.concatenate(all_labels, axis=0)
            labels = xyxy_pixels_to_xywh_norm(labels_xyxy, mosaic_size)
        else:
            labels = np.zeros((0, 5), dtype=np.float32)

        return canvas, labels, str(first_image_path)

    def __getitem__(self, index: int):
        if self.augment and random.random() < self.mosaic_prob:
            image, labels, image_path = self.load_mosaic(index)
        else:
            image, labels, image_path = self.load_raw_sample(index)

            if self.augment:
                image, labels = apply_train_augmentations(image, labels)

            image, labels = letterbox_image_and_labels(
                image=image,
                labels=labels,
                image_size=self.image_size,
            )

            image_path = str(image_path)

        image_np = np.array(image, dtype=np.float32) / 255.0

        image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).contiguous()
        target_tensor = torch.from_numpy(labels).float()

        return image_tensor, target_tensor, image_path


def detection_collate_fn(batch):
    images = []
    targets = []
    image_paths = []

    for image, target, image_path in batch:
        images.append(image)
        targets.append(target)
        image_paths.append(image_path)

    images = torch.stack(images, dim=0)

    return images, targets, image_paths