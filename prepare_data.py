"""
Chuẩn bị dataset PHÁT HIỆN LỖI BỀ MẶT (defect detection) - bộ NEU-DET.
6 loại lỗi thép: crazing, inclusion, patches, pitted_surface, rolled-in_scale, scratches.

Nguồn: https://github.com/Marfbin/NEU-DET-with-yolov8 (đã ở định dạng YOLO).
Tạo subset cân bằng để train nhanh trên Apple M1.

Chạy: python prepare_data.py
"""
import random
import shutil
import subprocess
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).parent
DATASET = ROOT / "dataset"
TMP = ROOT / ".tmp_neu"
REPO = "https://github.com/Marfbin/NEU-DET-with-yolov8.git"

PER_CLASS_TRAIN = 50   # ảnh/lớp cho train  -> 300 ảnh
PER_CLASS_VAL = 12     # ảnh/lớp cho val    -> 72 ảnh

NAMES = ["crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches"]


def get_source():
    """Clone repo (shallow) nếu chưa có, trả về thư mục NEU-DET."""
    neu = TMP / "data" / "NEU-DET"
    if not neu.exists():
        print("Tải dataset NEU-DET từ GitHub (shallow clone)...")
        subprocess.run(["git", "clone", "--depth", "1", REPO, str(TMP)], check=True)
    return neu


def class_of(stem):
    """Suy ra tên lớp từ tên file, vd 'rolled-in_scale_205' -> 'rolled-in_scale'."""
    for n in sorted(NAMES, key=len, reverse=True):
        if stem.startswith(n):
            return n
    return None


def build(src):
    img_dir = src / "train" / "images"
    lbl_dir = src / "train" / "labels"

    # gom ảnh theo lớp
    by_class = defaultdict(list)
    for img in sorted(img_dir.glob("*.jpg")):
        c = class_of(img.stem)
        if c and (lbl_dir / f"{img.stem}.txt").exists():
            by_class[c].append(img)

    if DATASET.exists():
        shutil.rmtree(DATASET)
    for split in ("train", "val"):
        (DATASET / "images" / split).mkdir(parents=True, exist_ok=True)
        (DATASET / "labels" / split).mkdir(parents=True, exist_ok=True)

    random.seed(42)
    totals = {"train": 0, "val": 0}
    for c, imgs in by_class.items():
        random.shuffle(imgs)
        picks = {
            "train": imgs[:PER_CLASS_TRAIN],
            "val": imgs[PER_CLASS_TRAIN:PER_CLASS_TRAIN + PER_CLASS_VAL],
        }
        for split, files in picks.items():
            for f in files:
                shutil.copy(f, DATASET / "images" / split / f.name)
                shutil.copy(lbl_dir / f"{f.stem}.txt", DATASET / "labels" / split / f"{f.stem}.txt")
                totals[split] += 1

    names_yaml = "\n".join(f"  {i}: {n}" for i, n in enumerate(NAMES))
    (DATASET / "data.yaml").write_text(
        f"path: {DATASET}\n"
        "train: images/train\n"
        "val: images/val\n"
        f"nc: {len(NAMES)}\n"
        f"names:\n{names_yaml}\n"
    )
    print(f"train: {totals['train']} ảnh | val: {totals['val']} ảnh")
    print(f"Đã ghi {DATASET / 'data.yaml'}")


if __name__ == "__main__":
    build(get_source())
    print("\n✅ Xong! Dataset lỗi bề mặt sẵn sàng tại ./dataset")
