# CIFAR-10 Data Pruning: EL2N & GraNd

Dự án thực nghiệm kỹ thuật cắt tỉa dữ liệu (Data Pruning / Coreset Selection) sử dụng **EL2N** (Error L2-Norm) và **GraNd** (Gradient Normed) trên tập dữ liệu CIFAR-10 với kiến trúc ResNet-18 (theo bài báo *Deep Learning on a Data Diet: Finding Important Examples Early in Training* - Paul et al., NeurIPS 2021).

---

## 📊 Kết quả Thực nghiệm Tổng hợp (24 Lượt Chạy Chính)

Kết quả trung bình (Mean ± Std) qua 3 random seeds (42, 43, 44) sau 150 epoch huấn luyện:

| Tỷ lệ Coreset | Số ảnh huấn luyện | EL2N Best Accuracy (%) | GraNd Best Accuracy (%) | EL2N Final Macro-F1 | GraNd Final Macro-F1 |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **10%** | **5.000 ảnh** | **80.46% ± 3.62%** | **79.17% ± 1.10%** | **0.7208** | **0.7889** |
| **5%** | **2.500 ảnh** | **66.36% ± 1.61%** | **68.86% ± 1.10%** | **0.6570** | **0.6880** |
| **3%** | **1.500 ảnh** | **56.21% ± 3.82%** | **59.35% ± 0.50%** | **0.5493** | **0.5978** |
| **1%** | **500 ảnh** | **37.34% ± 2.25%** | **49.23% ± 0.90%** | **0.3483** | **0.4908** |

*(So với Baseline 100% dữ liệu ban đầu đạt **92.27% ± 0.31%** sau 20 epoch warm-up).*

---

## ⚡ So sánh Hiệu năng Tính toán (Scoring Time)
* **EL2N:** **55.42 giây** cho 50.000 mẫu trên cả 3 seed (~18.5s / seed).
* **GraNd:** **585.69 giây** cho 50.000 mẫu trên cả 3 seed (~195.2s / seed).
* **Tương quan thứ hạng Spearman:** **0.9984** (xác nhận EL2N xấp xỉ thứ tự mẫu giống GraNd tới 99.84% nhưng chạy nhanh hơn **gấp 10.5 lần**).

---

## Cấu trúc thư mục
```text
CIFAR10_EL2N_GraNd/
├── configs/config.yaml           # Cấu hình siêu tham số
├── src/                          # Mã nguồn thực nghiệm
│   ├── data.py                   # Quản lý Dataset & DataLoader
│   ├── model.py                  # ResNet-18 chuẩn hóa 32x32
│   ├── warmup.py                 # Huấn luyện warm-up 20 epoch
│   ├── el2n.py                   # Tính điểm EL2N
│   ├── grand.py                  # Tính điểm GraNd (torch.func.vmap)
│   ├── coreset.py                # Lọc Coreset Top-k
│   ├── train.py                  # Huấn luyện trên Coreset
│   ├── run_experiments.py        # Tự động hóa thí nghiệm
│   ├── evaluate.py               # Đo Accuracy & Macro-F1
│   └── checkpoint.py             # Quản lý Checkpoint & Resume
├── results/                      # Lưu trữ kết quả thực nghiệm
│   └── summary_results.csv       # Bảng chi tiết toàn bộ các lượt chạy
├── requirements.txt
└── README.md
```
