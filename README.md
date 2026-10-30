# YOLO11s Detection — Project thử nghiệm

Nhận diện vật thể (object detection) bằng **YOLO11s** + **PyTorch**, chạy trên **Apple M1 (mps)**.
Train thử với ~40 ảnh lấy từ dataset mẫu COCO128.

## Cấu trúc

```
yolo/
├── .venv/              # môi trường Python 3.12 (đã cài ultralytics + torch)
├── dataset/            # 40 ảnh + nhãn (32 train / 8 val) + data.yaml
├── prepare_data.py     # tải dataset mẫu & tạo subset 40 ảnh
├── train.py            # train YOLO11s
├── predict.py          # nhận diện trên ảnh
├── requirements.txt
└── runs/               # kết quả train & predict (model, biểu đồ, ảnh)
```

## Cách dùng

Luôn dùng python trong venv: `.venv/bin/python`

```bash
# 1. (Đã chạy rồi) Tạo data 40 ảnh
.venv/bin/python prepare_data.py

# 2. Train
.venv/bin/python train.py

# 3. Nhận diện
.venv/bin/python predict.py                 # trên ảnh val mẫu
.venv/bin/python predict.py duong_dan.jpg   # trên ảnh của bạn
.venv/bin/python predict.py 0               # webcam (số 0)
```

## Kết quả lần train đầu
- Train 16 epoch (early stopping tự dừng), ~2.5 phút trên M1
- **mAP50 = 0.684**, **mAP50-95 = 0.433**
- Model lưu tại: `runs/train_thu/weights/best.pt`
- Ảnh nhận diện: `runs/.../predict/`

## Tinh chỉnh trong `train.py`
| Tham số | Ý nghĩa | Gợi ý |
|---|---|---|
| `model="yolo11s.pt"` | cỡ model | `n` nhẹ hơn, `m/l/x` chính xác hơn (cần GPU mạnh) |
| `epochs` | số vòng học | data nhiều thì tăng (100–300) |
| `patience` | tự dừng nếu không cải thiện | giữ 15–30 |
| `batch` | số ảnh mỗi lần | M1 nên để 8; OOM thì giảm |
| `device` | phần cứng | `mps` (M1) / `cpu` / `0` (GPU NVIDIA) |
| `imgsz` | kích thước ảnh | 640 chuẩn; vật nhỏ tăng 1280 |

## Dùng data của riêng bạn
1. Gán nhãn ảnh bằng [Roboflow](https://roboflow.com) hoặc [CVAT](https://cvat.ai), xuất **định dạng YOLO**.
2. Bỏ ảnh vào `dataset/images/{train,val}`, nhãn `.txt` vào `dataset/labels/{train,val}`.
3. Sửa `dataset/data.yaml` cho đúng `nc` (số class) và `names`.
4. Chạy lại `train.py`.
