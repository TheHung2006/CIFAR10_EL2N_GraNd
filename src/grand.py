import os
import time
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.func import functional_call, grad, vmap
from tqdm import tqdm

from data import get_scoring_loader
from model import get_model

def compute_loss(params, buffers, model, x, y):
    """Tính Cross-Entropy Loss cho 1 mẫu đơn lẻ."""
    out = functional_call(model, (params, buffers), x.unsqueeze(0))
    return F.cross_entropy(out, y.unsqueeze(0))

def compute_grand_for_one_model(model, scoring_loader, device="cpu", batch_size=64):
    """
    Tính điểm GraNd chuẩn lý thuyết cho 50.000 mẫu:
    GraNd(x_i) = ||∇_θ L(f_θ(x_i), y_i)||_2 (chuẩn L2 của gradient theo toàn bộ 11M tham số).
    Sử dụng torch.func.vmap để tính song song per-sample gradient tốc độ cao trên GPU.
    """
    model.eval()
    params = dict(model.named_parameters())
    buffers = dict(model.named_buffers())

    # Hàm tính gradient cho 1 mẫu: grad đối với params (argnums=0)
    def single_sample_loss(p, b, x, y):
        return compute_loss(p, b, model, x, y)

    # Vector hóa trên chiều batch của x và y bằng vmap
    per_sample_grad_fn = vmap(grad(single_sample_loss, argnums=0), in_dims=(None, None, 0, 0))

    total_samples = len(scoring_loader.dataset)
    scores = np.zeros(total_samples, dtype=np.float32)
    labels_all = np.zeros(total_samples, dtype=np.int64)

    pbar = tqdm(scoring_loader, desc="Tính GraNd", leave=False)
    for images, targets, indices in pbar:
        images = images.to(device)
        targets_gpu = targets.to(device)

        # Tính per-sample gradient cho toàn bộ batch
        per_sample_grads = per_sample_grad_fn(params, buffers, images, targets_gpu)

        # Tính chuẩn L2: ||g_i||_2 = sqrt( sum(g_k^2) )
        # Tổng bình phương gradient trên tất cả các layer
        norm_sq = sum(g.pow(2).flatten(start_dim=1).sum(dim=1) for g in per_sample_grads.values())
        batch_grand_norms = torch.sqrt(norm_sq)

        idx_np = indices.numpy()
        scores[idx_np] = batch_grand_norms.detach().cpu().numpy()
        labels_all[idx_np] = targets.numpy()

        # Giải phóng bộ nhớ GPU của batch
        del per_sample_grads, norm_sq, batch_grand_norms
        if device.type == "cuda":
            torch.cuda.empty_cache()

    return scores, labels_all

def compute_grand_all_seeds(seeds=[42, 43, 44], batch_size=64, data_dir="./data", warmup_dir="./results/warmup", save_dir="./results/scores"):
    os.makedirs(save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n==========================================")
    print(f"BẮT ĐẦU TÍNH ĐIỂM GraNd | DEVICE: {device}")
    print(f"Sử dụng torch.func.vmap trên toàn bộ tham số mạng")
    print(f"==========================================")

    scoring_loader = get_scoring_loader(data_dir=data_dir, batch_size=batch_size)
    total_samples = len(scoring_loader.dataset)
    print(f"Số lượng mẫu cần chấm điểm: {total_samples} | Batch size: {batch_size}")

    all_scores = {}
    labels = None

    start_time = time.time()

    for s in seeds:
        ckpt_path = os.path.join(warmup_dir, f"warmup_seed{s}.pth")
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Không tìm thấy checkpoint: {ckpt_path}")

        print(f"\n--> Đang tính điểm GraNd trên Seed {s}...")
        t0 = time.time()
        model = get_model(num_classes=10, device=device)
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])

        scores_s, labels_s = compute_grand_for_one_model(model, scoring_loader, device=device, batch_size=batch_size)
        all_scores[f"seed_{s}"] = scores_s
        if labels is None:
            labels = labels_s

        elapsed = time.time() - t0
        print(f"    ✓ Xong Seed {s} trong: {elapsed:.2f} giây (~{elapsed/60:.1f} phút)")

    total_grand_time = time.time() - start_time
    print(f"\n[Xong] Tổng thời gian tính GraNd cho 3 seed: {total_grand_time:.2f} giây (~{total_grand_time/60:.2f} phút)")

    # 1. Tính trung bình cộng score qua 3 seed
    score_matrix = np.column_stack([all_scores[f"seed_{s}"] for s in seeds])
    mean_scores = np.mean(score_matrix, axis=1)

    # 2. Lưu grand_scores.npy
    npy_path = os.path.join(save_dir, "grand_scores.npy")
    np.save(npy_path, mean_scores)
    print(f"✓ Đã lưu file numpy: {npy_path}")

    # 3. Lưu DataFrame CSV: index, label, score_seed..., score_mean
    df_data = {
        "index": np.arange(total_samples),
        "label": labels
    }
    for s in seeds:
        df_data[f"score_seed{s}"] = all_scores[f"seed_{s}"]
    df_data["score_mean"] = mean_scores

    df = pd.DataFrame(df_data)
    csv_path = os.path.join(save_dir, "grand_scores.csv")
    df.to_csv(csv_path, index=False)
    print(f"✓ Đã lưu file CSV chi tiết: {csv_path}")

    # 4. Lưu thời gian tính toán vào scoring_time.csv
    time_csv_path = os.path.join(save_dir, "scoring_time.csv")
    time_row = {
        "method": "GraNd",
        "num_samples": total_samples,
        "num_seeds": len(seeds),
        "total_time_sec": round(total_grand_time, 2),
        "avg_time_per_seed_sec": round(total_grand_time / len(seeds), 2)
    }
    if os.path.exists(time_csv_path):
        time_df = pd.read_csv(time_csv_path)
        time_df = pd.concat([time_df[time_df["method"] != "GraNd"], pd.DataFrame([time_row])])
    else:
        time_df = pd.DataFrame([time_row])
    time_df.to_csv(time_csv_path, index=False)
    print(f"✓ Đã cập nhật thời gian vào: {time_csv_path}")

    # Thống kê nhanh điểm số
    print(f"\n--- PHÂN BỐ ĐIỂM GraNd ---")
    print(f"Min score : {mean_scores.min():.4f}")
    print(f"Mean score: {mean_scores.mean():.4f}")
    print(f"Max score : {mean_scores.max():.4f}")

    # 5. So sánh tương quan với EL2N (nếu có file el2n_scores.npy)
    el2n_path = os.path.join(save_dir, "el2n_scores.npy")
    if os.path.exists(el2n_path):
        el2n_scores = np.load(el2n_path)
        from scipy.stats import spearmanr, pearsonr
        pearson_corr, _ = pearsonr(mean_scores, el2n_scores)
        spearman_corr, _ = spearmanr(mean_scores, el2n_scores)
        print(f"\n--- ĐỐI CHIẾU LÝ THUYẾT VỚI EL2N ---")
        print(f"Độ tương quan tuyến tính Pearson  (GraNd vs EL2N): {pearson_corr:.4f}")
        print(f"Độ tương quan thứ hạng Spearman (GraNd vs EL2N): {spearman_corr:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--warmup_dir", type=str, default="./results/warmup")
    parser.add_argument("--save_dir", type=str, default="./results/scores")
    args = parser.parse_args()

    compute_grand_all_seeds(
        seeds=args.seeds,
        batch_size=args.batch_size,
        data_dir=args.data_dir,
        warmup_dir=args.warmup_dir,
        save_dir=args.save_dir
    )
