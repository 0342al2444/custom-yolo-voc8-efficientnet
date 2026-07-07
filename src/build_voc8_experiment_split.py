from pathlib import Path
import shutil
from collections import defaultdict


IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]

VOC8_CLASS_NAMES = [
    "person",
    "car",
    "dog",
    "cat",
    "bus",
    "train",
    "bicycle",
    "aeroplane",
]

# Current dataset.py expects original VOC20 IDs in label files and remaps them
# internally to VOC8 IDs. Therefore this builder keeps original VOC20 label IDs.
VOC20_TO_VOC8 = {
    14: 0,  # person
    6: 1,   # car
    11: 2,  # dog
    7: 3,   # cat
    5: 4,   # bus
    18: 5,  # train
    1: 6,   # bicycle
    0: 7,   # aeroplane
}


def find_split_dirs(dataset_root: Path, split: str):
    candidates = [
        (
            dataset_root / "images" / split,
            dataset_root / "labels" / split,
        ),
        (
            dataset_root / split / "images",
            dataset_root / split / "labels",
        ),
    ]

    for image_dir, label_dir in candidates:
        if image_dir.exists() and label_dir.exists():
            return image_dir, label_dir

    return None, None


def list_images(image_dir: Path):
    image_paths = []

    if not image_dir.exists():
        return image_paths

    for ext in IMAGE_EXTENSIONS:
        image_paths.extend(image_dir.glob(f"*{ext}"))

    return sorted(image_paths)


def find_image_for_label(image_dir: Path, label_path: Path):
    for ext in IMAGE_EXTENSIONS:
        image_path = image_dir / f"{label_path.stem}{ext}"

        if image_path.exists():
            return image_path

    return None


def safe_copy(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def safe_write_text(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def read_label_lines(label_path: Path):
    if not label_path.exists():
        return []

    text = label_path.read_text(encoding="utf-8").strip()

    if not text:
        return []

    return [line.strip() for line in text.splitlines() if line.strip()]


def count_labels(label_dir: Path):
    object_count = 0
    class_counts = defaultdict(int)
    unknown_class_count = 0

    for label_path in sorted(label_dir.glob("*.txt")):
        lines = read_label_lines(label_path)

        for line in lines:
            parts = line.split()

            if len(parts) < 5:
                continue

            class_id = int(float(parts[0]))

            if class_id not in VOC20_TO_VOC8:
                unknown_class_count += 1
                continue

            voc8_id = VOC20_TO_VOC8[class_id]

            class_counts[voc8_id] += 1
            object_count += 1

    return object_count, class_counts, unknown_class_count


def copy_split(
    source_root: Path,
    source_split: str,
    output_root: Path,
    output_split: str,
    prefix: str,
):
    source_image_dir, source_label_dir = find_split_dirs(
        dataset_root=source_root,
        split=source_split,
    )

    if source_image_dir is None or source_label_dir is None:
        raise FileNotFoundError(
            f"Could not find split '{source_split}' in {source_root}"
        )

    output_image_dir = output_root / "images" / output_split
    output_label_dir = output_root / "labels" / output_split

    label_paths = sorted(source_label_dir.glob("*.txt"))

    copied_images = 0
    copied_labels = 0
    missing_images = 0

    for label_path in label_paths:
        image_path = find_image_for_label(
            image_dir=source_image_dir,
            label_path=label_path,
        )

        if image_path is None:
            missing_images += 1
            continue

        new_stem = f"{prefix}_{label_path.stem}"

        output_image_path = output_image_dir / f"{new_stem}{image_path.suffix.lower()}"
        output_label_path = output_label_dir / f"{new_stem}.txt"

        safe_copy(image_path, output_image_path)
        safe_copy(label_path, output_label_path)

        copied_images += 1
        copied_labels += 1

    object_count, _, unknown_class_count = count_labels(output_label_dir)

    print()
    print("-" * 80)
    print(f"Copied {prefix} {source_split} -> {output_split}")
    print("-" * 80)
    print(f"Source image dir: {source_image_dir}")
    print(f"Source label dir: {source_label_dir}")
    print(f"Output image dir: {output_image_dir}")
    print(f"Output label dir: {output_label_dir}")
    print(f"Copied images:    {copied_images}")
    print(f"Copied labels:    {copied_labels}")
    print(f"Missing images:   {missing_images}")
    print(f"Objects so far in output split '{output_split}': {object_count}")
    print(f"Unknown/non-VOC8 label IDs found: {unknown_class_count}")

    return {
        "source_root": source_root,
        "source_split": source_split,
        "output_split": output_split,
        "prefix": prefix,
        "copied_images": copied_images,
        "copied_labels": copied_labels,
        "missing_images": missing_images,
    }


def summarize_output_split(output_root: Path, split: str):
    image_dir, label_dir = find_split_dirs(
        dataset_root=output_root,
        split=split,
    )

    if image_dir is None or label_dir is None:
        print()
        print(f"[SKIP] Output split not found: {split}")
        return None

    image_count = len(list_images(image_dir))
    label_count = len(sorted(label_dir.glob("*.txt")))
    object_count, class_counts, unknown_class_count = count_labels(label_dir)

    print()
    print("=" * 80)
    print(f"Output split summary: {split}")
    print("=" * 80)
    print(f"Images:  {image_count}")
    print(f"Labels:  {label_count}")
    print(f"Objects: {object_count}")
    print(f"Unknown/non-VOC8 label IDs found: {unknown_class_count}")
    print()
    print("Class distribution:")

    for class_id, class_name in enumerate(VOC8_CLASS_NAMES):
        count = class_counts.get(class_id, 0)
        print(f"  {class_id}: {class_name:10s} {count}")

    return {
        "split": split,
        "images": image_count,
        "labels": label_count,
        "objects": object_count,
        "unknown_class_count": unknown_class_count,
        "class_counts": dict(class_counts),
    }


def write_classes_file(output_root: Path):
    safe_write_text(
        output_root / "classes.txt",
        "\n".join(VOC8_CLASS_NAMES) + "\n",
    )


def write_split_readme(output_root: Path):
    readme = """VOC2007 + VOC2012 custom VOC-8 split

train:
  VOC2007 train
  VOC2007 val
  VOC2012 train

val:
  VOC2007 test

test:
  VOC2012 val

Labels:
  Label files keep original VOC20 class IDs.
  dataset.py remaps original VOC20 IDs to VOC8 IDs while loading.

VOC8 class order:
  0 person
  1 car
  2 dog
  3 cat
  4 bus
  5 train
  6 bicycle
  7 aeroplane
"""
    safe_write_text(output_root / "SPLIT_README.txt", readme)


def main():
    project_root = Path(__file__).resolve().parents[1]

    voc2007_root = project_root / "data" / "processed" / "voc_yolo_voc8_trimmed"
    voc2012_root = project_root / "data" / "processed" / "voc2012_yolo_voc8_trimmed"

    output_root = project_root / "data" / "processed" / "voc2007_2012_custom_voc8"

    print("Build VOC2007 + VOC2012 Custom VOC-8 Split")
    print("=" * 80)
    print(f"VOC2007 root: {voc2007_root}")
    print(f"VOC2012 root: {voc2012_root}")
    print(f"Output root:  {output_root}")
    print()
    print("This script does NOT start training.")
    print()
    print("Requested split:")
    print("  train = VOC2007 train + VOC2007 val + VOC2012 train")
    print("  val   = VOC2007 test")
    print("  test  = VOC2012 val")
    print()
    print("Label policy:")
    print("  Keep original VOC20 class IDs in label files.")
    print("  dataset.py will remap them to VOC8 IDs when loading.")

    if not voc2007_root.exists():
        raise FileNotFoundError(f"VOC2007 trimmed dataset not found: {voc2007_root}")

    if not voc2012_root.exists():
        raise FileNotFoundError(f"VOC2012 trimmed dataset not found: {voc2012_root}")

    if output_root.exists():
        print()
        print("=" * 80)
        print("Removing old output folder")
        print("=" * 80)
        print(f"Old output folder: {output_root}")
        shutil.rmtree(output_root)

    output_root.mkdir(parents=True, exist_ok=True)

    copy_jobs = [
        (voc2007_root, "train", output_root, "train", "voc2007_train"),
        (voc2007_root, "val", output_root, "train", "voc2007_val"),
        (voc2012_root, "train", output_root, "train", "voc2012_train"),
        (voc2007_root, "test", output_root, "val", "voc2007_test"),
        (voc2012_root, "val", output_root, "test", "voc2012_val"),
    ]

    for source_root, source_split, dst_root, output_split, prefix in copy_jobs:
        copy_split(
            source_root=source_root,
            source_split=source_split,
            output_root=dst_root,
            output_split=output_split,
            prefix=prefix,
        )

    write_classes_file(output_root)
    write_split_readme(output_root)

    train_stats = summarize_output_split(output_root, "train")
    val_stats = summarize_output_split(output_root, "val")
    test_stats = summarize_output_split(output_root, "test")

    print()
    print("=" * 80)
    print("Final actual dataset usage")
    print("=" * 80)
    print("No training has been started.")
    print()

    print("Training split: train")
    print(f"Training images:  {train_stats['images']}")
    print(f"Training objects: {train_stats['objects']}")
    print()

    print("Validation split: val")
    print(f"Validation images:  {val_stats['images']}")
    print(f"Validation objects: {val_stats['objects']}")
    print()

    print("Final test split: test")
    print(f"Test images:  {test_stats['images']}")
    print(f"Test objects: {test_stats['objects']}")
    print()

    print("Update dataset_root to:")
    print(f"dataset_root = project_root / \"data\" / \"processed\" / \"{output_root.name}\"")
    print()
    print("Recommended split settings:")
    print('train_split = "train"')
    print('val_split = "val"')
    print()
    print("For final testing with eval.py, use:")
    print('split = "test"')
    print()
    print("Done.")


if __name__ == "__main__":
    main()