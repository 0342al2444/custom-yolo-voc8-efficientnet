from pathlib import Path
import shutil


VOC8_CLASSES = [
    "person",
    "car",
    "dog",
    "cat",
    "bus",
    "train",
    "bicycle",
    "aeroplane",
]

# Our custom dataset label files still use original VOC20 class IDs.
# Ultralytics needs class IDs to be 0 to 7.
OLD_VOC20_TO_NEW_VOC8 = {
    14: 0,  # person
    6: 1,   # car
    11: 2,  # dog
    7: 3,   # cat
    5: 4,   # bus
    18: 5,  # train
    1: 6,   # bicycle
    0: 7,   # aeroplane
}


def find_split_dirs(root: Path, split: str):
    image_dir = root / "images" / split
    label_dir = root / "labels" / split

    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")

    if not label_dir.exists():
        raise FileNotFoundError(f"Label directory not found: {label_dir}")

    return image_dir, label_dir


def convert_label_lines(source_label_path: Path):
    converted_lines = []
    skipped_lines = 0

    if not source_label_path.exists():
        return converted_lines, skipped_lines

    lines = source_label_path.read_text(encoding="utf-8").splitlines()

    for line in lines:
        line = line.strip()

        if not line:
            continue

        parts = line.split()

        if len(parts) != 5:
            skipped_lines += 1
            continue

        old_class_id = int(float(parts[0]))

        if old_class_id not in OLD_VOC20_TO_NEW_VOC8:
            skipped_lines += 1
            continue

        new_class_id = OLD_VOC20_TO_NEW_VOC8[old_class_id]

        x_center = parts[1]
        y_center = parts[2]
        width = parts[3]
        height = parts[4]

        converted_lines.append(
            f"{new_class_id} {x_center} {y_center} {width} {height}"
        )

    return converted_lines, skipped_lines


def copy_and_convert_split(
    source_root: Path,
    target_root: Path,
    split: str,
):
    source_image_dir, source_label_dir = find_split_dirs(source_root, split)

    target_image_dir = target_root / "images" / split
    target_label_dir = target_root / "labels" / split

    target_image_dir.mkdir(parents=True, exist_ok=True)
    target_label_dir.mkdir(parents=True, exist_ok=True)

    image_extensions = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]

    image_paths = [
        path
        for path in source_image_dir.iterdir()
        if path.is_file() and path.suffix.lower() in image_extensions
    ]

    image_paths = sorted(image_paths)

    copied_image_count = 0
    copied_object_count = 0
    skipped_line_count = 0
    missing_label_count = 0
    empty_after_remap_count = 0

    class_counts = {class_name: 0 for class_name in VOC8_CLASSES}

    for source_image_path in image_paths:
        source_label_path = source_label_dir / f"{source_image_path.stem}.txt"

        if not source_label_path.exists():
            missing_label_count += 1
            continue

        converted_lines, skipped_lines = convert_label_lines(source_label_path)

        skipped_line_count += skipped_lines

        # Important fairness rule:
        # If an image has no valid VOC8 labels after conversion,
        # do not copy the image into the YOLOv8n dataset.
        if len(converted_lines) == 0:
            empty_after_remap_count += 1
            continue

        target_image_path = target_image_dir / source_image_path.name
        target_label_path = target_label_dir / f"{source_image_path.stem}.txt"

        shutil.copy2(source_image_path, target_image_path)

        target_label_path.write_text(
            "\n".join(converted_lines) + "\n",
            encoding="utf-8",
        )

        copied_image_count += 1
        copied_object_count += len(converted_lines)

        for line in converted_lines:
            class_id = int(line.split()[0])
            class_counts[VOC8_CLASSES[class_id]] += 1

    return {
        "split": split,
        "images": copied_image_count,
        "objects": copied_object_count,
        "missing_labels": missing_label_count,
        "empty_after_remap": empty_after_remap_count,
        "skipped_lines": skipped_line_count,
        "class_counts": class_counts,
    }


def write_yaml(target_root: Path):
    yaml_path = target_root / "voc8.yaml"

    root_posix = target_root.resolve().as_posix()

    names_lines = []

    for class_id, class_name in enumerate(VOC8_CLASSES):
        names_lines.append(f"  {class_id}: {class_name}")

    yaml_text = (
        f"path: {root_posix}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"test: images/test\n"
        f"\n"
        f"nc: {len(VOC8_CLASSES)}\n"
        f"names:\n"
        + "\n".join(names_lines)
        + "\n"
    )

    yaml_path.write_text(yaml_text, encoding="utf-8")

    return yaml_path


def verify_ultralytics_dataset(target_root: Path):
    print()
    print("=" * 80)
    print("Verifying Ultralytics dataset")
    print("=" * 80)

    for split in ["train", "val", "test"]:
        image_dir = target_root / "images" / split
        label_dir = target_root / "labels" / split

        image_paths = sorted(image_dir.glob("*.*"))
        label_paths = sorted(label_dir.glob("*.txt"))

        empty_labels = []
        invalid_class_ids = []

        for label_path in label_paths:
            lines = label_path.read_text(encoding="utf-8").splitlines()

            if len(lines) == 0:
                empty_labels.append(label_path)

            for line in lines:
                parts = line.split()

                if len(parts) != 5:
                    invalid_class_ids.append((label_path, line))
                    continue

                class_id = int(float(parts[0]))

                if class_id < 0 or class_id >= len(VOC8_CLASSES):
                    invalid_class_ids.append((label_path, line))

        print(f"{split}:")
        print(f"  images: {len(image_paths)}")
        print(f"  labels: {len(label_paths)}")
        print(f"  empty label files: {len(empty_labels)}")
        print(f"  invalid class lines: {len(invalid_class_ids)}")

        if len(image_paths) != len(label_paths):
            print("  WARNING: image count and label count do not match.")

        if empty_labels:
            print("  WARNING: empty label files found.")

        if invalid_class_ids:
            print("  WARNING: invalid class IDs found.")

        print()


def main():
    project_root = Path(__file__).resolve().parents[1]

    # This source should already be the fair trimmed VOC8 split:
    # train = VOC2007 train + VOC2007 val + VOC2012 train
    # val   = VOC2007 test
    # test  = VOC2012 val
    source_root = project_root / "data" / "processed" / "voc2007_2012_custom_voc8"

    # This target is for Ultralytics YOLOv8n.
    # It contains the same trimmed images, but labels are remapped to 0 to 7.
    target_root = project_root / "data" / "processed" / "voc2007_2012_custom_voc8_ultralytics"

    print(f"Source root: {source_root}")
    print(f"Target root: {target_root}")
    print()

    if not source_root.exists():
        raise FileNotFoundError(
            f"Source dataset not found: {source_root}\n"
            "Run this first:\n"
            "  python src\\build_voc8_experiment_split.py"
        )

    if target_root.exists():
        print("Removing old Ultralytics dataset copy...")
        shutil.rmtree(target_root)

    target_root.mkdir(parents=True, exist_ok=True)

    summaries = []

    for split in ["train", "val", "test"]:
        print(f"Converting split: {split}")

        summary = copy_and_convert_split(
            source_root=source_root,
            target_root=target_root,
            split=split,
        )

        summaries.append(summary)

    yaml_path = write_yaml(target_root)

    print()
    print("=" * 80)
    print("Ultralytics VOC8 dataset created")
    print("=" * 80)
    print(f"Dataset root: {target_root}")
    print(f"YAML file:    {yaml_path}")
    print()

    for summary in summaries:
        print(f"{summary['split']}:")
        print(f"  copied images:      {summary['images']}")
        print(f"  copied objects:     {summary['objects']}")
        print(f"  missing labels:     {summary['missing_labels']}")
        print(f"  empty after remap:  {summary['empty_after_remap']}")
        print(f"  skipped lines:      {summary['skipped_lines']}")

        for class_name, count in summary["class_counts"].items():
            print(f"    {class_name:10s}: {count}")

        print()

    print("Expected counts if everything matches our trimmed custom split:")
    print("  train images: 7777, objects: 17122")
    print("  val images:   3665, objects: 7693")
    print("  test images:  4189, objects: 9187")
    print()

    verify_ultralytics_dataset(target_root)

    print("Done.")


if __name__ == "__main__":
    main()