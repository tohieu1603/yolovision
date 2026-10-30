"""
Train YOLO11s PHÁT HIỆN LỖI BỀ MẶT (defect detection) trên NEU-DET.
Dùng PyTorch + GPU Apple M1 ('mps').
Chạy: python train.py
"""
from pathlib import Path
from ultralytics import YOLO

ROOT = Path(__file__).parent


def main():
    model = YOLO("yolo11s.pt")   # pretrained cỡ small, tự tải lần đầu

    model.train(
        data=str(ROOT / "dataset" / "data.yaml"),
        epochs=80,          # lỗi bề mặt khó hơn -> train nhiều hơn
        patience=20,        # tự dừng nếu 20 epoch không cải thiện
        imgsz=640,
        batch=8,            # M1 -> batch nhỏ tránh hết RAM
        device="mps",
        workers=2,
        project=str(ROOT / "runs"),
        name="defect",
        exist_ok=True,
    )

    metrics = model.val()
    print("\n=== KẾT QUẢ ===")
    print(f"mAP50-95: {metrics.box.map:.3f}")
    print(f"mAP50   : {metrics.box.map50:.3f}")
    print(f"Model lưu tại: {ROOT / 'runs' / 'defect' / 'weights' / 'best.pt'}")


if __name__ == "__main__":
    main()
