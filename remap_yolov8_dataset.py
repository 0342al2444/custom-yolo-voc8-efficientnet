from pathlib import Path
import shutil

src = Path("data/processed/voc2007_2012_custom_voc8")
dst = Path("external_models/yolov8_fair_current_split/datasets/voc8_remapped")

# original VOC20 id -> new VOC8 id
mapping = {
    14: 0,  # person
    6: 1,   # car
    11: 2,  # dog
    7: 3,   # cat
    5: 4,   # bus
    18: 5,  # train
    1: 6,   # bicycle
    0: 7,   # aeroplane
}

for split in ["train", "val", "test"]:
    (dst / "images" / split).mkdir(parents=True, exist_ok=True)
    (dst / "labels" / split).mkdir(parents=True, exist_ok=True)

    for img in (src / "images" / split).iterdir():
        if img.is_file():
            shutil.copy2(img, dst / "images" / split / img.name)

    for lab in (src / "labels" / split).glob("*.txt"):
        out_lines = []

        for line in lab.read_text().splitlines():
            parts = line.strip().split()
            if not parts:
                continue

            old_class = int(float(parts[0]))

            if old_class in mapping:
                parts[0] = str(mapping[old_class])
                out_lines.append(" ".join(parts))

        output_text = "\n".join(out_lines)
        if output_text:
            output_text += "\n"

        (dst / "labels" / split / lab.name).write_text(output_text, encoding="utf-8")

print("Done:", dst)
