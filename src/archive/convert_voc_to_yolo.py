from pathlib import Path
import shutil
import xml.etree.ElementTree as ET


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

CLASS_TO_ID = {class_name: idx for idx, class_name in enumerate(VOC_CLASSES)}

# For a first clean training setup, skip objects marked difficult.
# Difficult objects are usually unclear, tiny, occluded, or hard to detect.
SKIP_DIFFICULT = True


def voc_box_to_yolo(
    xmin: float,
    ymin: float,
    xmax: float,
    ymax: float,
    image_width: float,
    image_height: float,
) -> tuple[float, float, float, float]:
    """
    Convert VOC box format to YOLO format.

    VOC format:
        xmin, ymin, xmax, ymax

    YOLO format:
        x_center_norm, y_center_norm, width_norm, height_norm

    All YOLO values are normalized to [0, 1].
    """
    x_center = (xmin + xmax) / 2.0
    y_center = (ymin + ymax) / 2.0
    box_width = xmax - xmin
    box_height = ymax - ymin

    x_center_norm = x_center / image_width
    y_center_norm = y_center / image_height
    width_norm = box_width / image_width
    height_norm = box_height / image_height

    return x_center_norm, y_center_norm, width_norm, height_norm


def clamp_box(
    xmin: float,
    ymin: float,
    xmax: float,
    ymax: float,
    image_width: float,
    image_height: float,
) -> tuple[float, float, float, float]:
    """
    Keep bounding boxes inside image boundaries.
    This prevents invalid boxes if an annotation touches or slightly exceeds the border.
    """
    xmin = max(0.0, min(xmin, image_width))
    ymin = max(0.0, min(ymin, image_height))
    xmax = max(0.0, min(xmax, image_width))
    ymax = max(0.0, min(ymax, image_height))
    return xmin, ymin, xmax, ymax


def convert_one_annotation(xml_path: Path, label_path: Path) -> int:
    """
    Convert one VOC XML annotation file into one YOLO TXT label file.

    Returns:
        number of objects written to the label file.
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    size_node = root.find("size")
    if size_node is None:
        raise ValueError(f"Missing size node in {xml_path}")

    image_width = float(size_node.findtext("width"))
    image_height = float(size_node.findtext("height"))

    lines = []

    for obj in root.findall("object"):
        class_name = obj.findtext("name")
        difficult = int(obj.findtext("difficult", default="0"))

        if class_name not in CLASS_TO_ID:
            continue

        if SKIP_DIFFICULT and difficult == 1:
            continue

        box = obj.find("bndbox")
        if box is None:
            continue

        xmin = float(box.findtext("xmin"))
        ymin = float(box.findtext("ymin"))
        xmax = float(box.findtext("xmax"))
        ymax = float(box.findtext("ymax"))

        xmin, ymin, xmax, ymax = clamp_box(
            xmin, ymin, xmax, ymax, image_width, image_height
        )

        if xmax <= xmin or ymax <= ymin:
            continue

        x_center, y_center, width, height = voc_box_to_yolo(
            xmin, ymin, xmax, ymax, image_width, image_height
        )

        class_id = CLASS_TO_ID[class_name]

        line = (
            f"{class_id} "
            f"{x_center:.6f} "
            f"{y_center:.6f} "
            f"{width:.6f} "
            f"{height:.6f}"
        )
        lines.append(line)

    label_path.parent.mkdir(parents=True, exist_ok=True)
    label_path.write_text("\n".join(lines), encoding="utf-8")

    return len(lines)


def read_split_ids(split_file: Path) -> list[str]:
    """
    Read image IDs from VOC split files.

    Example:
        train.txt contains lines like:
        000012
        000017
    """
    return [
        line.strip()
        for line in split_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def convert_split(
    split_name: str,
    voc_root: Path,
    output_root: Path,
) -> tuple[int, int]:
    """
    Convert one split: train, val, or test.

    Copies images and creates YOLO label files.
    """
    split_file = voc_root / "ImageSets" / "Main" / f"{split_name}.txt"
    image_ids = read_split_ids(split_file)

    image_src_dir = voc_root / "JPEGImages"
    annotation_src_dir = voc_root / "Annotations"

    image_out_dir = output_root / "images" / split_name
    label_out_dir = output_root / "labels" / split_name

    image_out_dir.mkdir(parents=True, exist_ok=True)
    label_out_dir.mkdir(parents=True, exist_ok=True)

    total_objects = 0

    for image_id in image_ids:
        image_src = image_src_dir / f"{image_id}.jpg"
        xml_src = annotation_src_dir / f"{image_id}.xml"

        image_dst = image_out_dir / f"{image_id}.jpg"
        label_dst = label_out_dir / f"{image_id}.txt"

        if not image_src.exists():
            raise FileNotFoundError(f"Missing image: {image_src}")

        if not xml_src.exists():
            raise FileNotFoundError(f"Missing annotation: {xml_src}")

        shutil.copy2(image_src, image_dst)
        objects_written = convert_one_annotation(xml_src, label_dst)
        total_objects += objects_written

    return len(image_ids), total_objects


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]

    voc_root = project_root / "data" / "raw" / "VOCdevkit" / "VOC2007"
    output_root = project_root / "data" / "processed" / "voc_yolo"

    if not voc_root.exists():
        raise FileNotFoundError(f"VOC folder not found: {voc_root}")

    print(f"VOC root: {voc_root}")
    print(f"Output root: {output_root}")
    print()

    for split_name in ["train", "val", "test"]:
        num_images, num_objects = convert_split(split_name, voc_root, output_root)
        print(
            f"{split_name}: "
            f"{num_images} images, "
            f"{num_objects} objects written"
        )

    classes_file = output_root / "classes.txt"
    classes_file.write_text("\n".join(VOC_CLASSES), encoding="utf-8")

    print()
    print("Done.")
    print(f"Classes saved to: {classes_file}")


if __name__ == "__main__":
    main()