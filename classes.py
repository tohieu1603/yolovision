"""
Xem danh sách các lớp (class) của một model .pt
Chạy:
    python classes.py                         # xem model đã train (best.pt)
    python classes.py yolo11s.pt              # xem model pretrained gốc
    python classes.py duong_dan/model.pt      # xem model bất kỳ
"""
import sys
from pathlib import Path
from ultralytics import YOLO

ROOT = Path(__file__).parent
DEFAULT = ROOT / "runs" / "train_thu" / "weights" / "best.pt"
if not DEFAULT.exists():
    DEFAULT = ROOT / "runs" / "detect" / "runs" / "train_thu" / "weights" / "best.pt"


def main():
    weights = sys.argv[1] if len(sys.argv) > 1 else str(DEFAULT)
    model = YOLO(weights)
    print(f"Model : {weights}")
    print(f"Số lớp: {len(model.names)}\n")
    for i, name in model.names.items():
        print(f"{i:>3}: {name}")


if __name__ == "__main__":
    main()
