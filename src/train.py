import os
import time
import json
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from data import get_train_loader, get_test_loader
from model import get_model
from evaluate import evaluate
from checkpoint import set_seed, save_checkpoint, load_checkpoint, get_random_states

def train_coreset_run(
    method="el2n",
    size=5000,
    seed=42,
    warmup_epochs=20,
    total_epochs=150,
    lr=0.05,
    batch_size=128,
    data_dir="./data",
    warmup_dir="./results/warmup",
    coresets_dir="./results/coresets",
    checkpoints_base="./results/checkpoints",
    resume=True
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_name = f"{method}_{size}_seed{seed}"
    run_dir = os.path.join(checkpoints_base, run_name)
    os.makedirs(run_dir, exist_ok=True)

    print(f"\n==========================================")
    print(f"CHẠY HUẤN LUYỆN CORESET: {run_name.upper()}")
    print(f"Phương pháp: {method} | Số ảnh: {size} | Seed: {seed}")
    print(f"Tiến trình: từ epoch {warmup_epochs + 1} đến epoch {total_epochs}")
    print(f"Device: {device}")
    print(f"==========================================")

    # 1. Cố định seed
    set_seed(seed)

    # 2. Nạp danh sách index coreset
    if method == "full":
        subset_indices = None
        print(f"--> Sử dụng toàn bộ 50.000 ảnh (Full Data)")
    else:
        coreset_file = os.path.join(coresets_dir, f"{method.lower()}_coreset_{size}.npy")
        if not os.path.exists(coreset_file):
            raise FileNotFoundError(f"Không tìm thấy file coreset: {coreset_file}")
        subset_indices = np.load(coreset_file)
        print(f"--> Đã nạp {len(subset_indices)} index từ: {coreset_file}")

    # Đảm bảo batch_size không lớn hơn kích thước coreset
    effective_batch_size = min(batch_size, size) if size > 0 else batch_size
    train_loader = get_train_loader(data_dir=data_dir, batch_size=effective_batch_size, subset_indices=subset_indices)
    test_loader = get_test_loader(data_dir=data_dir, batch_size=128)

    # 3. Khởi tạo mô hình
    model = get_model(num_classes=10, device=device)
    criterion = nn.CrossEntropyLoss()

    # 4. Nạp checkpoint Warm-up tại epoch 20
    warmup_pth = os.path.join(warmup_dir, f"warmup_seed{seed}.pth")
    if not os.path.exists(warmup_pth):
        raise FileNotFoundError(f"Không tìm thấy checkpoint warm-up: {warmup_pth}")

    warmup_ckpt = torch.load(warmup_pth, map_location=device)
    model.load_state_dict(warmup_ckpt["model_state_dict"])
    print(f"✓ Đã nạp thành công trọng số warm-up epoch {warmup_epochs} từ {warmup_pth}")

    # Optimizer và Scheduler cho giai đoạn fine-tune tiếp theo
    remaining_epochs = total_epochs - warmup_epochs
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=remaining_epochs)

    # 5. Kiểm tra resume nếu có lần chạy dở dang
    latest_ckpt = os.path.join(run_dir, "latest.pth")
    start_epoch = warmup_epochs
    history = []
    best_acc = 0.0

    if resume and os.path.exists(latest_ckpt):
        start_epoch, history, best_acc = load_checkpoint(
            latest_ckpt, model, optimizer=optimizer, scheduler=scheduler, device=device
        )
        print(f"[Resume] Tiếp tục từ epoch {start_epoch + 1}/{total_epochs}")

    if start_epoch >= total_epochs:
        print(f"[Done] Lượt chạy {run_name} đã hoàn thành đủ {total_epochs} epoch từ trước!")
        return

    # Lưu run_config.json
    config_path = os.path.join(run_dir, "run_config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump({
            "run_name": run_name,
            "method": method,
            "coreset_size": size,
            "seed": seed,
            "warmup_epochs": warmup_epochs,
            "total_epochs": total_epochs,
            "learning_rate": lr,
            "effective_batch_size": effective_batch_size
        }, f, indent=2, ensure_ascii=False)

    # 6. Vòng lặp huấn luyện từ epoch 21 đến total_epochs
    metrics_file = os.path.join(run_dir, "metrics.csv")
    total_start_time = time.time()

    for epoch in range(start_epoch, total_epochs):
        epoch_start = time.time()
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, targets in train_loader:
            images, targets = images.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)

        train_loss = running_loss / total
        train_acc = (correct / total) * 100.0

        scheduler.step()

        # Đánh giá trên tập test 10.000 ảnh
        test_loss, test_acc, test_f1 = evaluate(model, test_loader, criterion=criterion, device=device)
        epoch_time = time.time() - epoch_start

        row_metric = {
            "epoch": epoch + 1,
            "train_loss": round(train_loss, 4),
            "train_accuracy": round(train_acc, 2),
            "test_loss": round(test_loss, 4),
            "test_accuracy": round(test_acc, 2),
            "test_macro_f1": round(test_f1, 4),
            "epoch_time_sec": round(epoch_time, 2),
            "cumulative_time_sec": round(time.time() - total_start_time, 2)
        }
        history.append(row_metric)

        print(f"Epoch [{epoch+1:03d}/{total_epochs:03d}] "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
              f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}% | "
              f"Macro-F1: {test_f1:.4f} | Time: {epoch_time:.2f}s")

        # Lưu latest.pth
        ckpt_state = {
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "random_states": get_random_states(),
            "history": history,
            "best_acc": max(best_acc, test_acc)
        }
        save_checkpoint(ckpt_state, run_dir, filename="latest.pth")

        # Lưu best.pth nếu đạt độ chính xác cao nhất
        if test_acc > best_acc:
            best_acc = test_acc
            save_checkpoint(ckpt_state, run_dir, filename="best.pth")

        # Cập nhật metrics.csv
        pd.DataFrame(history).to_csv(metrics_file, index=False)

    # 7. Lưu final.pth khi hoàn tất
    final_pth = os.path.join(run_dir, "final.pth")
    torch.save({
        "epoch": total_epochs,
        "model_state_dict": model.state_dict(),
        "final_test_acc": test_acc,
        "final_macro_f1": test_f1,
        "best_acc": best_acc
    }, final_pth)

    print(f"\n✅ Hoàn thành lượt chạy {run_name}!")
    print(f"Best Test Acc: {best_acc:.2f}% | Final Test Acc: {test_acc:.2f}% | Final Macro-F1: {test_f1:.4f}")

def main():
    parser = argparse.ArgumentParser(description="Train on Coreset")
    parser.add_argument("--method", type=str, default="el2n", choices=["el2n", "grand", "random", "full"])
    parser.add_argument("--size", type=int, default=5000, help="Số lượng ảnh trong coreset")
    parser.add_argument("--seed", type=int, default=42, help="Seed của warm-up model")
    parser.add_argument("--warmup_epochs", type=int, default=20)
    parser.add_argument("--total_epochs", type=int, default=25, help="Tổng epoch (mặc định 25 cho chạy thử, 150 cho chính thức)")
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--warmup_dir", type=str, default="./results/warmup")
    parser.add_argument("--coresets_dir", type=str, default="./results/coresets")
    parser.add_argument("--checkpoints_base", type=str, default="./results/checkpoints")
    args = parser.parse_args()

    train_coreset_run(
        method=args.method,
        size=args.size,
        seed=args.seed,
        warmup_epochs=args.warmup_epochs,
        total_epochs=args.total_epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        data_dir=args.data_dir,
        warmup_dir=args.warmup_dir,
        coresets_dir=args.coresets_dir,
        checkpoints_base=args.checkpoints_base
    )

if __name__ == "__main__":
    main()
