import os
import time
import argparse
import pandas as pd
import torch

from train import train_coreset_run

def run_all_experiments(
    methods=["el2n", "grand"],
    sizes=[5000],
    seeds=[42, 43, 44],
    total_epochs=150,
    warmup_epochs=20,
    lr=0.05,
    batch_size=128,
    data_dir="./data",
    warmup_dir="./results/warmup",
    coresets_dir="./results/coresets",
    checkpoints_base="./results/checkpoints",
    summary_file="./results/summary_results.csv"
):
    os.makedirs(checkpoints_base, exist_ok=True)
    os.makedirs(os.path.dirname(summary_file), exist_ok=True)

    # Đọc summary cũ nếu đã có (để append tiếp)
    if os.path.exists(summary_file):
        summary_df = pd.read_csv(summary_file)
    else:
        summary_df = pd.DataFrame(columns=[
            "method", "coreset_size", "seed", "best_test_acc", "final_test_acc", "final_macro_f1", "status"
        ])

    total_runs = len(methods) * len(sizes) * len(seeds)
    run_idx = 0

    print("==================================================")
    print("BẮT ĐẦU CHẠY THỰC NGHIỆM TỰ ĐỘNG (AUTOMATED RUNNER)")
    print(f"Danh sách phương pháp : {methods}")
    print(f"Danh sách kích thước  : {sizes}")
    print(f"Danh sách seed        : {seeds}")
    print(f"Số epoch mỗi lượt     : {warmup_epochs} -> {total_epochs}")
    print(f"Tổng số lượt chạy     : {total_runs}")
    print("==================================================")

    for method in methods:
        for size in sizes:
            for seed in seeds:
                run_idx += 1
                run_name = f"{method}_{size}_seed{seed}"
                run_dir = os.path.join(checkpoints_base, run_name)
                final_pth = os.path.join(run_dir, "final.pth")

                # Kiểm tra xem lượt chạy này đã hoàn thành trước đó chưa
                if os.path.exists(final_pth):
                    try:
                        ckpt = torch.load(final_pth, map_location="cpu", weights_only=False)
                    except TypeError:
                        ckpt = torch.load(final_pth, map_location="cpu")
                    if ckpt.get("epoch", 0) >= total_epochs:
                        print(f"\n[Bỏ qua] Lượt chạy {run_name} ({run_idx}/{total_runs}) đã hoàn thành đủ {total_epochs} epoch từ trước!")
                        continue

                print(f"\n>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
                print(f"[{run_idx}/{total_runs}] TIẾN HÀNH CHẠY: {run_name.upper()}")
                print(f">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")

                t_start = time.time()
                train_coreset_run(
                    method=method,
                    size=size,
                    seed=seed,
                    warmup_epochs=warmup_epochs,
                    total_epochs=total_epochs,
                    lr=lr,
                    batch_size=batch_size,
                    data_dir=data_dir,
                    warmup_dir=warmup_dir,
                    coresets_dir=coresets_dir,
                    checkpoints_base=checkpoints_base,
                    resume=True
                )
                elapsed = time.time() - t_start

                # Đọc kết quả từ final.pth để cập nhật vào summary_results.csv
                if os.path.exists(final_pth):
                    try:
                        ckpt = torch.load(final_pth, map_location="cpu", weights_only=False)
                    except TypeError:
                        ckpt = torch.load(final_pth, map_location="cpu")
                    best_acc = ckpt.get("best_acc", ckpt.get("final_test_acc", 0.0))
                    final_acc = ckpt.get("final_test_acc", 0.0)
                    final_f1 = ckpt.get("final_macro_f1", 0.0)

                    new_row = {
                        "method": method.upper(),
                        "coreset_size": size,
                        "seed": seed,
                        "best_test_acc": round(float(best_acc), 2),
                        "final_test_acc": round(float(final_acc), 2),
                        "final_macro_f1": round(float(final_f1), 4),
                        "training_time_sec": round(elapsed, 2),
                        "status": "COMPLETED"
                    }

                    # Xóa dòng cũ nếu trùng run_name rồi thêm dòng mới
                    if len(summary_df) > 0:
                        mask = (summary_df["method"] == method.upper()) & (summary_df["coreset_size"] == size) & (summary_df["seed"] == seed)
                        summary_df = summary_df[~mask]

                    summary_df = pd.concat([summary_df, pd.DataFrame([new_row])], ignore_index=True)
                    summary_df.to_csv(summary_file, index=False)
                    print(f"✓ Đã cập nhật kết quả vào: {summary_file}")

    print("\n==================================================")
    print("HOÀN THÀNH TẤT CẢ CÁC LƯỢT CHẠY!")
    print("==================================================")
    if os.path.exists(summary_file):
        df = pd.read_csv(summary_file)
        print("\n--- BẢNG TỔNG HỢP KẾT QUẢ HIỆN TẠI ---")
        print(df.to_string(index=False))

def main():
    parser = argparse.ArgumentParser(description="Automated Runner for Coreset Experiments")
    parser.add_argument("--methods", type=str, nargs="+", default=["el2n", "grand"])
    parser.add_argument("--sizes", type=int, nargs="+", default=[5000])
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--total_epochs", type=int, default=150)
    parser.add_argument("--warmup_epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--batch_size", type=int, default=128)
    args = parser.parse_args()

    run_all_experiments(
        methods=args.methods,
        sizes=args.sizes,
        seeds=args.seeds,
        total_epochs=args.total_epochs,
        warmup_epochs=args.warmup_epochs,
        lr=args.lr,
        batch_size=args.batch_size
    )

if __name__ == "__main__":
    main()
