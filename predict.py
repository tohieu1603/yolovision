"""
Dùng model đã train để PHÁT HIỆN LỖI trên ảnh sản phẩm.
Chạy:
    python predict.py                  # chạy trên ảnh val có sẵn
    python predict.py duong_dan.jpg    # chạy trên ảnh của bạn
"""
import sys
from pathlib import Path
from ultralytics import YOLO

ROOT = Path(__file__).parent
BEST = ROOT / "runs" / "defect" / "weights" / "best.pt"


def main():
    weights = BEST if BEST.exists() else "yolo11s.pt"
    print(f"Dùng model: {weights}")
    model = YOLO(str(weights))

    source = sys.argv[1] if len(sys.argv) > 1 else str(ROOT / "dataset" / "images" / "val")

    results = model.predict(
        source=source,
        save=True,
        conf=0.25,
        device="mps",
        project=str(ROOT / "runs"),
        name="predict",
        exist_ok=True,
    )
    print(f"\n✅ Xong! Ảnh kết quả lưu tại: {ROOT / 'runs' / 'predict'}")
    for r in results:
        print(f"  {Path(r.path).name}: phát hiện {len(r.boxes)} lỗi")


if __name__ == "__main__":
    main()
