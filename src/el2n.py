import os
import time
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from data import get_scoring_loader
from model import get_model

def compute_el2n_for_one_model(model, scoring_loader, device="cpu"):
    """
    Tính điểm EL2N cho 50.000 mẫu trên 1 checkpoint:
    EL2N(x_i) = ||p(x_i) - y_i||_2 (chuẩn Euclidean của vector sai số xác suất).
    """
    model.eval()
    scores = np.zeros(len(scoring_loader.dataset), dtype=np.float32)
    labels_all = np.zeros(len(scoring_loader.dataset), dtype=np.int64)

    with torch.no_grad():
        for images, targets, indices in scoring_loader:
            images = images.to(device)
            targets_gpu = targets.to(device)

            # 1. Forward pass lấy logits
            logits = model(images)
            # 2. Xác suất Softmax: p(x_i)
            probs = F.softmax(logits, dim=1)

            # 3. Vector one-hot nhãn thực tế
            one_hot = F.one_hot(targets_gpu, num_classes=10).float()

            # 4. Vector sai số e_i = p_i - y_i
            error_vectors = probs - one_hot

            # 5. Chuẩn L2 của vector sai số: ||e_i||_2
            batch_l2_norms = torch.norm(error_vectors, p=2, dim=1)

            # Gán vào mảng kết quả theo đúng index của mẫu
            idx_np = indices.numpy()
            scores[idx_np] = batch_l2_norms.cpu().numpy()
            labels_all[idx_np] = targets.numpy()

    return scores, labels_all

def compute_el2n_all_seeds(seeds=[42, 43, 44], data_dir="./data", warmup_dir="./results/warmup", save_dir="./results/scores"):
    os.makedirs(save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n==========================================")
    print(f"BẮT ĐẦU TÍNH ĐIỂM EL2N | DEVICE: {device}")
    print(f"==========================================")

    # 1. Scoring loader: KHÔNG Augmentation, giữ nguyên index
    scoring_loader = get_scoring_loader(data_dir=data_dir, batch_size=256)
    total_samples = len(scoring_loader.dataset)
    print(f"Số lượng mẫu cần chấm điểm: {total_samples}")

    all_scores = {}
    labels = None

    start_time = time.time()

    for s in seeds:
        ckpt_path = os.path.join(warmup_dir, f"warmup_seed{s}.pth")
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Không tìm thấy checkpoint: {ckpt_path}")

        print(f"--> Đang tính điểm EL2N trên Seed {s}...")
        t0 = time.time()
        model = get_model(num_classes=10, device=device)
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])

        scores_s, labels_s = compute_el2n_for_one_model(model, scoring_loader, device=device)
        all_scores[f"seed_{s}"] = scores_s
        if labels is None:
            labels = labels_s

        print(f"    Xong Seed {s} trong: {time.time() - t0:.2f} giây")

    total_el2n_time = time.time() - start_time
    print(f"\n[Xong] Tổng thời gian tính EL2N cho 3 seed: {total_el2n_time:.2f} giây")

    # 2. Tính điểm trung bình cộng qua 3 seed
    score_matrix = np.column_stack([all_scores[f"seed_{s}"] for s in seeds])
    mean_scores = np.mean(score_matrix, axis=1)

    # 3. Lưu el2n_scores.npy
    npy_path = os.path.join(save_dir, "el2n_scores.npy")
    np.save(npy_path, mean_scores)
    print(f"✓ Đã lưu file numpy: {npy_path}")

    # 4. Lưu DataFrame CSV: index, label, score của từng seed và mean score
    df_data = {
        "index": np.arange(total_samples),
        "label": labels
    }
    for s in seeds:
        df_data[f"score_seed{s}"] = all_scores[f"seed_{s}"]
    df_data["score_mean"] = mean_scores

    df = pd.DataFrame(df_data)
    csv_path = os.path.join(save_dir, "el2n_scores.csv")
    df.to_csv(csv_path, index=False)
    print(f"✓ Đã lưu file CSV chi tiết: {csv_path}")

    # 5. Lưu thời gian tính toán vào scoring_time.csv
    time_csv_path = os.path.join(save_dir, "scoring_time.csv")
    time_row = {
        "method": "EL2N",
        "num_samples": total_samples,
        "num_seeds": len(seeds),
        "total_time_sec": round(total_el2n_time, 2),
        "avg_time_per_seed_sec": round(total_el2n_time / len(seeds), 2)
    }
    if os.path.exists(time_csv_path):
        time_df = pd.read_csv(time_csv_path)
        time_df = pd.concat([time_df[time_df["method"] != "EL2N"], pd.DataFrame([time_row])])
    else:
        time_df = pd.DataFrame([time_row])
    time_df.to_csv(time_csv_path, index=False)
    print(f"✓ Đã cập nhật thời gian vào: {time_csv_path}")

    # Thống kê nhanh điểm số
    print(f"\n--- PHÂN BỐ ĐIỂM EL2N ---")
    print(f"Min score : {mean_scores.min():.4f} (Mẫu cực dễ, mô hình đoán tự tin 100%)")
    print(f"Mean score: {mean_scores.mean():.4f}")
    print(f"Max score : {mean_scores.max():.4f} (Mẫu khó / Outlier / Nhãn nhiễu)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--warmup_dir", type=str, default="./results/warmup")
    parser.add_argument("--save_dir", type=str, default="./results/scores")
    args = parser.parse_args()

    compute_el2n_all_seeds(
        seeds=args.seeds,
        data_dir=args.data_dir,
        warmup_dir=args.warmup_dir,
        save_dir=args.save_dir
    )
