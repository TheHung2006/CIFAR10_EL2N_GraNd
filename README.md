# CIFAR-10 Data Pruning: EL2N & GraNd

Dự án thực nghiệm kỹ thuật cắt tỉa dữ liệu (Data Pruning / Coreset Selection) sử dụng **EL2N** (Error L2-Norm) và **GraNd** (Gradient Normed) trên tập dữ liệu CIFAR-10 với kiến trúc ResNet-18.

## Cấu trúc thư mục
```text
CIFAR10_EL2N_GraNd/
├── configs/config.yaml     # File cấu hình tham số thực nghiệm
├── src/                    # Mã nguồn chính
│   ├── data.py             # Quản lý Dataset và DataLoader
│   ├── model.py            # Mô hình ResNet18 (chuẩn hóa 32x32)
│   ├── warmup.py           # Huấn luyện warm-up 20 epoch (3 seed)
│   ├── el2n.py             # Tính điểm EL2N
│   ├── grand.py            # Tính điểm GraNd
│   ├── coreset.py          # Lọc và chọn Coreset theo tỷ lệ
│   ├── train.py            # Huấn luyện tiếp trên Coreset
│   ├── evaluate.py         # Đánh giá Accuracy & Macro-F1
│   └── checkpoint.py       # Quản lý checkpoint & resume
├── notebooks/              # Các notebook chạy thực nghiệm trên Kaggle
├── results/                # Lưu trữ kết quả (weights, scores, metrics, figures)
├── requirements.txt
└── README.md
```
