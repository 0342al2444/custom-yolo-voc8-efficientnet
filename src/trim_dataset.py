from pathlib import Path
import shutil
from collections import defaultdict


IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]

# Original Pascal VOC 20-class order used in the label files.
VOC20_CLASS_NAMES = [
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

# Our selected 8 classes, in the same order used by the model.
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

# Original VOC20 class id -> new VOC8 class id.
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

SELECTED_VOC20_IDS = set(VOC20_TO_VOC8.keys())

# IMPORTANT:
# Keep this False if your dataset.py already remaps original VOC20 IDs to our VOC8 IDs.
# Based on your current training/eval working correctly, this should stay False.
#
# If later you change dataset.py to expect label files already using 0-7 class IDs,
# then set this to True.
REMAP_LABEL_IDS_TO_VOC8 = False

# These should match your current train.py settings.
TRAIN_SPLIT_ACTUALLY_USED = "trainval"
VAL_SPLIT_ACTUALLY_USED = "test"


def list_images(image_dir: Path):
    image_paths = []

    if not image_dir.exists():
        return image_paths

    for ext in IMAGE_EXTENSIONS:
        image_paths.extend(image_dir.glob(f"*{ext}"))

    return sorted(image_paths)


def find_split_dirs(dataset_root: Path, split: str):
    candidates = [
        (
            dataset_root / "images" / split,
            dataset_root / "labels" / split,
            "layout_a",
        ),
        (
            dataset_root / split / "images",
            dataset_root / split / "labels",
            "layout_b",
        ),
    ]

    for image_dir, label_dir, layout_name in candidates:
        if image_dir.exists() and label_dir.exists():
            return image_dir, label_dir, layout_name

    return None, None, None


def find_image_for_label(image_dir: Path, label_path: Path):
    stem = label_path.stem

    for ext in IMAGE_EXTENSIONS:
        candidate = image_dir / f"{stem}{ext}"

        if candidate.exists():
            return candidate

    return None


def read_label_lines(label_path: Path):
    if not label_path.exists():
        return []

    text = label_path.read_text(encoding="utf-8").strip()

    if not text:
        return []

    return [line.strip() for line in text.splitlines() if line.strip()]


def parse_label_line(line: str):
    parts = line.split()

    if len(parts) < 5:
        return None

    try:
        class_id = int(float(parts[0]))
    except ValueError:
        return None

    return class_id, parts


def filter_selected_voc8_lines(label_path: Path):
    """
    Keeps only objects from our selected 8 classes.

    By default, output label class IDs stay as original VOC20 IDs.
    This keeps compatibility with your current dataset.py, which appears to
    already filter/remap VOC20 IDs internally.

    If REMAP_LABEL_IDS_TO_VOC8 = True, output labels will use 0-7 class IDs.
    """

    kept_lines = []
    kept_original_counts = defaultdict(int)
    kept_voc8_counts = defaultdict(int)
    removed_other_objects = 0
    malformed_lines = 0

    lines = read_label_lines(label_path)

    for line in lines:
        parsed = parse_label_line(line)

        if parsed is None:
            malformed_lines += 1
            continue

        original_class_id, parts = parsed

        if original_class_id not in SELECTED_VOC20_IDS:
            removed_other_objects += 1
            continue

        voc8_class_id = VOC20_TO_VOC8[original_class_id]

        kept_original_counts[original_class_id] += 1
        kept_voc8_counts[voc8_class_id] += 1

        if REMAP_LABEL_IDS_TO_VOC8:
            parts[0] = str(voc8_class_id)
        else:
            parts[0] = str(original_class_id)

        kept_lines.append(" ".join(parts))

    return {
        "kept_lines": kept_lines,
        "kept_original_counts": dict(kept_original_counts),
        "kept_voc8_counts": dict(kept_voc8_counts),
        "removed_other_objects": removed_other_objects,
        "malformed_lines": malformed_lines,
        "total_lines": len(lines),
    }


def safe_copy(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def safe_write_text(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def merge_count_dict(target, source):
    for key, value in source.items():
        target[key] += value


def count_one_existing_split(dataset_root: Path, split: str):
    image_dir, label_dir, layout_name = find_split_dirs(
        dataset_root=dataset_root,
        split=split,
    )

    if image_dir is None or label_dir is None:
        return {
            "split": split,
            "found": False,
            "image_dir": None,
            "label_dir": None,
            "layout": None,
            "image_files": 0,
            "label_files": 0,
            "images_with_any_label": 0,
            "images_with_selected_voc8_label": 0,
            "objects_all_classes": 0,
            "objects_selected_voc8": 0,
            "objects_removed_other_classes": 0,
            "malformed_lines": 0,
            "voc8_counts": {},
            "original_counts": {},
        }

    image_paths = list_images(image_dir)
    label_paths = sorted(label_dir.glob("*.txt"))

    image_stems = {path.stem for path in image_paths}

    images_with_any_label = 0
    images_with_selected_voc8_label = 0
    objects_all_classes = 0
    objects_selected_voc8 = 0
    objects_removed_other_classes = 0
    malformed_lines = 0

    voc8_counts = defaultdict(int)
    original_counts = defaultdict(int)

    for label_path in label_paths:
        if label_path.stem not in image_stems:
            continue

        lines = read_label_lines(label_path)

        if lines:
            images_with_any_label += 1

        objects_all_classes += len(lines)

        filtered = filter_selected_voc8_lines(label_path)

        if filtered["kept_lines"]:
            images_with_selected_voc8_label += 1

        objects_selected_voc8 += len(filtered["kept_lines"])
        objects_removed_other_classes += filtered["removed_other_objects"]
        malformed_lines += filtered["malformed_lines"]

        merge_count_dict(voc8_counts, filtered["kept_voc8_counts"])
        merge_count_dict(original_counts, filtered["kept_original_counts"])

    return {
        "split": split,
        "found": True,
        "image_dir": image_dir,
        "label_dir": label_dir,
        "layout": layout_name,
        "image_files": len(image_paths),
        "label_files": len(label_paths),
        "images_with_any_label": images_with_any_label,
        "images_with_selected_voc8_label": images_with_selected_voc8_label,
        "objects_all_classes": objects_all_classes,
        "objects_selected_voc8": objects_selected_voc8,
        "objects_removed_other_classes": objects_removed_other_classes,
        "malformed_lines": malformed_lines,
        "voc8_counts": dict(voc8_counts),
        "original_counts": dict(original_counts),
    }


def combine_stats_for_trainval(dataset_root: Path):
    train_stats = count_one_existing_split(dataset_root, "train")
    val_stats = count_one_existing_split(dataset_root, "val")

    if not train_stats["found"] or not val_stats["found"]:
        return count_one_existing_split(dataset_root, "trainval")

    voc8_counts = defaultdict(int)
    original_counts = defaultdict(int)

    merge_count_dict(voc8_counts, train_stats["voc8_counts"])
    merge_count_dict(voc8_counts, val_stats["voc8_counts"])

    merge_count_dict(original_counts, train_stats["original_counts"])
    merge_count_dict(original_counts, val_stats["original_counts"])

    return {
        "split": "trainval",
        "found": True,
        "image_dir": "combined train + val",
        "label_dir": "combined train + val",
        "layout": "combined",
        "image_files": train_stats["image_files"] + val_stats["image_files"],
        "label_files": train_stats["label_files"] + val_stats["label_files"],
        "images_with_any_label": (
            train_stats["images_with_any_label"]
            + val_stats["images_with_any_label"]
        ),
        "images_with_selected_voc8_label": (
            train_stats["images_with_selected_voc8_label"]
            + val_stats["images_with_selected_voc8_label"]
        ),
        "objects_all_classes": (
            train_stats["objects_all_classes"]
            + val_stats["objects_all_classes"]
        ),
        "objects_selected_voc8": (
            train_stats["objects_selected_voc8"]
            + val_stats["objects_selected_voc8"]
        ),
        "objects_removed_other_classes": (
            train_stats["objects_removed_other_classes"]
            + val_stats["objects_removed_other_classes"]
        ),
        "malformed_lines": (
            train_stats["malformed_lines"]
            + val_stats["malformed_lines"]
        ),
        "voc8_counts": dict(voc8_counts),
        "original_counts": dict(original_counts),
    }


def count_split_usage(dataset_root: Path, split: str):
    if split == "trainval":
        direct_stats = count_one_existing_split(dataset_root, "trainval")

        if direct_stats["found"]:
            return direct_stats

        return combine_stats_for_trainval(dataset_root)

    return count_one_existing_split(dataset_root, split)


def print_split_usage_report(dataset_root: Path, split: str):
    stats = count_split_usage(
        dataset_root=dataset_root,
        split=split,
    )

    print()
    print("-" * 80)
    print(f"Split: {split}")
    print("-" * 80)

    if not stats["found"]:
        print("Split not found.")
        return stats

    print(f"Image dir: {stats['image_dir']}")
    print(f"Label dir: {stats['label_dir']}")
    print(f"Layout:    {stats['layout']}")
    print()
    print(f"Image files found:                         {stats['image_files']}")
    print(f"Label files found:                         {stats['label_files']}")
    print(f"Images with any VOC20 label:               {stats['images_with_any_label']}")
    print(f"Images with selected VOC-8 labels:         {stats['images_with_selected_voc8_label']}")
    print(f"Objects from all VOC20 classes:            {stats['objects_all_classes']}")
    print(f"Objects kept from selected VOC-8 classes:  {stats['objects_selected_voc8']}")
    print(f"Objects removed from other 12 classes:     {stats['objects_removed_other_classes']}")
    print(f"Malformed label lines:                     {stats['malformed_lines']}")

    print()
    print("Selected VOC-8 distribution in model class order:")

    for class_id, class_name in enumerate(VOC8_CLASS_NAMES):
        count = stats["voc8_counts"].get(class_id, 0)
        print(f"  {class_id}: {class_name:10s} {count}")

    print()
    print("Selected VOC-8 original VOC20 IDs:")

    for original_id, voc8_id in sorted(VOC20_TO_VOC8.items()):
        class_name = VOC20_CLASS_NAMES[original_id]
        count = stats["original_counts"].get(original_id, 0)
        print(
            f"  VOC20 id {original_id:2d} -> VOC8 id {voc8_id}: "
            f"{class_name:10s} {count}"
        )

    return stats


def print_training_validation_usage(dataset_root: Path, title: str):
    print()
    print("=" * 80)
    print(title)
    print("=" * 80)
    print(f"Dataset root: {dataset_root}")
    print()
    print("This is only a count check.")
    print("No training process is started.")
    print()
    print(f"Training split used by train.py:   {TRAIN_SPLIT_ACTUALLY_USED}")
    print(f"Validation split used by train.py: {VAL_SPLIT_ACTUALLY_USED}")

    train_stats = print_split_usage_report(
        dataset_root=dataset_root,
        split=TRAIN_SPLIT_ACTUALLY_USED,
    )

    val_stats = print_split_usage_report(
        dataset_root=dataset_root,
        split=VAL_SPLIT_ACTUALLY_USED,
    )

    print()
    print("=" * 80)
    print("Actual training and validation image count")
    print("=" * 80)

    if train_stats["found"]:
        print(
            f"Actual training images with selected VOC-8 labels:   "
            f"{train_stats['images_with_selected_voc8_label']}"
        )
    else:
        print("Actual training images with selected VOC-8 labels:   split not found")

    if val_stats["found"]:
        print(
            f"Actual validation images with selected VOC-8 labels: "
            f"{val_stats['images_with_selected_voc8_label']}"
        )
    else:
        print("Actual validation images with selected VOC-8 labels: split not found")

    print()
    print("Note:")
    print("This count is based on the selected 8 classes only.")
    print("Images that contain only the other 12 VOC classes are not useful for our VOC-8 detector.")

    return train_stats, val_stats


def trim_one_split(source_root: Path, output_root: Path, split: str):
    image_dir, label_dir, layout_name = find_split_dirs(
        dataset_root=source_root,
        split=split,
    )

    if image_dir is None or label_dir is None:
        print(f"[SKIP] Split not found: {split}")
        return {
            "split": split,
            "found": False,
            "kept_images": 0,
            "removed_images_without_selected_classes": 0,
            "objects_kept": 0,
            "objects_removed_other_classes": 0,
            "missing_images": 0,
            "total_label_files": 0,
            "voc8_counts": {},
        }

    label_paths = sorted(label_dir.glob("*.txt"))

    kept_images = 0
    removed_images_without_selected_classes = 0
    objects_kept = 0
    objects_removed_other_classes = 0
    missing_images = 0
    malformed_lines = 0
    total_label_files = 0

    voc8_counts = defaultdict(int)

    print()
    print("=" * 80)
    print(f"Trimming split: {split}")
    print("=" * 80)
    print(f"Source image dir: {image_dir}")
    print(f"Source label dir: {label_dir}")
    print(f"Detected layout:  {layout_name}")
    print(f"Output label IDs remapped to VOC8 0-7: {REMAP_LABEL_IDS_TO_VOC8}")

    for label_path in label_paths:
        total_label_files += 1

        image_path = find_image_for_label(
            image_dir=image_dir,
            label_path=label_path,
        )

        if image_path is None:
            missing_images += 1
            continue

        filtered = filter_selected_voc8_lines(label_path)

        objects_removed_other_classes += filtered["removed_other_objects"]
        malformed_lines += filtered["malformed_lines"]

        if not filtered["kept_lines"]:
            removed_images_without_selected_classes += 1
            continue

        output_image_path = output_root / "images" / split / image_path.name
        output_label_path = output_root / "labels" / split / label_path.name

        safe_copy(image_path, output_image_path)

        safe_write_text(
            output_label_path,
            "\n".join(filtered["kept_lines"]) + "\n",
        )

        kept_images += 1
        objects_kept += len(filtered["kept_lines"])

        merge_count_dict(voc8_counts, filtered["kept_voc8_counts"])

    print()
    print(f"Total label files checked:                 {total_label_files}")
    print(f"Kept images with selected VOC-8 objects:   {kept_images}")
    print(f"Removed images without selected classes:   {removed_images_without_selected_classes}")
    print(f"Kept selected VOC-8 objects:               {objects_kept}")
    print(f"Removed other-class objects:               {objects_removed_other_classes}")
    print(f"Missing images:                            {missing_images}")
    print(f"Malformed label lines:                     {malformed_lines}")

    print()
    print("Selected VOC-8 object distribution:")

    for class_id, class_name in enumerate(VOC8_CLASS_NAMES):
        count = voc8_counts.get(class_id, 0)
        print(f"  {class_id}: {class_name:10s} {count}")

    return {
        "split": split,
        "found": True,
        "kept_images": kept_images,
        "removed_images_without_selected_classes": removed_images_without_selected_classes,
        "objects_kept": objects_kept,
        "objects_removed_other_classes": objects_removed_other_classes,
        "missing_images": missing_images,
        "malformed_lines": malformed_lines,
        "total_label_files": total_label_files,
        "voc8_counts": dict(voc8_counts),
    }


def create_trainval_from_train_and_val(output_root: Path):
    train_image_dir = output_root / "images" / "train"
    train_label_dir = output_root / "labels" / "train"

    val_image_dir = output_root / "images" / "val"
    val_label_dir = output_root / "labels" / "val"

    if not train_image_dir.exists() or not train_label_dir.exists():
        print()
        print("[SKIP] Cannot create trainval because trimmed train split was not found.")
        return

    if not val_image_dir.exists() or not val_label_dir.exists():
        print()
        print("[SKIP] Cannot create trainval because trimmed val split was not found.")
        return

    trainval_image_dir = output_root / "images" / "trainval"
    trainval_label_dir = output_root / "labels" / "trainval"

    trainval_image_dir.mkdir(parents=True, exist_ok=True)
    trainval_label_dir.mkdir(parents=True, exist_ok=True)

    copied_images = 0
    copied_labels = 0

    for split_name in ["train", "val"]:
        image_dir = output_root / "images" / split_name
        label_dir = output_root / "labels" / split_name

        for image_path in list_images(image_dir):
            safe_copy(
                image_path,
                trainval_image_dir / image_path.name,
            )
            copied_images += 1

        for label_path in sorted(label_dir.glob("*.txt")):
            safe_copy(
                label_path,
                trainval_label_dir / label_path.name,
            )
            copied_labels += 1

    print()
    print("=" * 80)
    print("Created trimmed trainval split")
    print("=" * 80)
    print(f"Images copied into trainval: {copied_images}")
    print(f"Labels copied into trainval: {copied_labels}")
    print(f"Trainval image dir: {trainval_image_dir}")
    print(f"Trainval label dir: {trainval_label_dir}")


def write_classes_file(output_root: Path):
    classes_path = output_root / "classes.txt"

    safe_write_text(
        classes_path,
        "\n".join(VOC8_CLASS_NAMES) + "\n",
    )

    print()
    print(f"Wrote classes file: {classes_path}")


def main():
    project_root = Path(__file__).resolve().parents[1]

    source_root = project_root / "data" / "processed" / "voc_yolo"
    output_root = project_root / "data" / "processed" / "voc_yolo_voc8_trimmed"

    splits_to_trim = ["train", "val", "test"]

    print("VOC-8 Dataset Trimmer")
    print("=" * 80)
    print(f"Source root: {source_root}")
    print(f"Output root: {output_root}")
    print()
    print("This script removes images that do not contain our selected 8 classes.")
    print("It also removes label lines from the other 12 VOC classes.")
    print("It does NOT delete your original dataset.")
    print("It does NOT start training.")
    print()
    print("Selected VOC-8 classes:")
    for new_id, class_name in enumerate(VOC8_CLASS_NAMES):
        original_id = None

        for old_id, mapped_id in VOC20_TO_VOC8.items():
            if mapped_id == new_id:
                original_id = old_id
                break

        print(f"  VOC8 id {new_id}: {class_name:10s} from original VOC20 id {original_id}")

    print()
    print(f"Output label IDs remapped to VOC8 0-7: {REMAP_LABEL_IDS_TO_VOC8}")
    print()

    if not source_root.exists():
        raise FileNotFoundError(f"Source dataset not found: {source_root}")

    print_training_validation_usage(
        dataset_root=source_root,
        title="Before trimming: current source dataset usage",
    )

    if output_root.exists():
        print()
        print("=" * 80)
        print("Removing old trimmed output folder")
        print("=" * 80)
        print(f"Old output folder: {output_root}")
        shutil.rmtree(output_root)

    output_root.mkdir(parents=True, exist_ok=True)

    results = []

    for split in splits_to_trim:
        result = trim_one_split(
            source_root=source_root,
            output_root=output_root,
            split=split,
        )

        results.append(result)

    create_trainval_from_train_and_val(output_root=output_root)
    write_classes_file(output_root=output_root)

    print()
    print("=" * 80)
    print("Trim summary")
    print("=" * 80)

    total_kept_images = 0
    total_removed_images = 0
    total_objects_kept = 0
    total_objects_removed = 0
    total_missing_images = 0

    for result in results:
        if not result["found"]:
            continue

        total_kept_images += result["kept_images"]
        total_removed_images += result["removed_images_without_selected_classes"]
        total_objects_kept += result["objects_kept"]
        total_objects_removed += result["objects_removed_other_classes"]
        total_missing_images += result["missing_images"]

        print(
            f"{result['split']:8s} | "
            f"kept images {result['kept_images']:5d} | "
            f"removed images {result['removed_images_without_selected_classes']:5d} | "
            f"kept objects {result['objects_kept']:5d} | "
            f"removed objects {result['objects_removed_other_classes']:5d}"
        )

    print("-" * 80)
    print(f"Total kept images:              {total_kept_images}")
    print(f"Total removed images:           {total_removed_images}")
    print(f"Total kept selected objects:    {total_objects_kept}")
    print(f"Total removed other objects:    {total_objects_removed}")
    print(f"Total missing images:           {total_missing_images}")

    print_training_validation_usage(
        dataset_root=output_root,
        title="After trimming: actual dataset usage if train.py uses trimmed dataset",
    )

    print()
    print("=" * 80)
    print("Done")
    print("=" * 80)
    print()
    print("Next step:")
    print("To actually train with the trimmed dataset, update dataset_root in train.py, eval.py, and predict.py to:")
    print()
    print(f"dataset_root = project_root / \"data\" / \"processed\" / \"{output_root.name}\"")
    print()
    print("This script only trims and counts. It does not start training.")


if __name__ == "__main__":
    main()