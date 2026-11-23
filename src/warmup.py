import os
import sys
import time
import json
import argparse
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from data import get_train_loader, get_test_loader
from model import get_model
from evaluate import evaluate
from checkpoint import set_seed, save_checkpoint, load_checkpoint, get_random_states

def train_warmup_one_seed(seed, epochs=20, lr=0.1, batch_size=128, data_dir="./data", save_dir="./results/warmup", resume=True):
    """
    Huấn luyện Warm-up cho 1 seed:
    - 20 epoch trên 50.000 ảnh CIFAR-10.
    - Lưu checkpoint đầy đủ sau mỗi epoch để chống ngắt kết nối Kaggle.
    - Ghi lại warmup_metrics.csv.
    """
    os.makedirs(save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n==========================================")
    print(f"BẮT ĐẦU WARM-UP SEED {seed} | DEVICE: {device}")
    print(f"==========================================")

    # 1. Cố định seed
    set_seed(seed)

    # 2. Dữ liệu
    train_loader = get_train_loader(data_dir=data_dir, batch_size=batch_size)
    test_loader = get_test_loader(data_dir=data_dir, batch_size=batch_size)

    # 3. Mô hình, hàm Loss, Optimizer, Scheduler
    model = get_model(num_classes=10, device=device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # 4. Kiểm tra resume nếu có checkpoint dở dang
    latest_ckpt_path = os.path.join(save_dir, f"warmup_seed{seed}_latest.pth")
    start_epoch = 0
    history = []
    best_acc = 0.0

    if resume and os.path.exists(latest_ckpt_path):
        start_epoch, history, best_acc = load_checkpoint(
            latest_ckpt_path, model, optimizer=optimizer, scheduler=scheduler, device=device
        )
        print(f"[Resume] Tiếp tục huấn luyện seed {seed} từ epoch {start_epoch + 1}/{epochs}")

    if start_epoch >= epochs:
        print(f"[Done] Seed {seed} đã hoàn thành đủ {epochs} epoch từ trước!")
        return

    # Lưu warmup_config.json
    config_path = os.path.join(save_dir, "warmup_config.json")
    if not os.path.exists(config_path):
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump({
                "model": "ResNet18 (32x32 modified)",
                "dataset": "CIFAR-10",
                "epochs": epochs,
                "learning_rate": lr,
                "momentum": 0.9,
                "weight_decay": 0.0005,
                "batch_size": batch_size,
                "seeds": [42, 43, 44]
            }, f, indent=2, ensure_ascii=False)

    # 5. Vòng lặp huấn luyện
    metrics_file = os.path.join(save_dir, "warmup_metrics.csv")
    total_start_time = time.time()

    for epoch in range(start_epoch, epochs):
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

        # Cập nhật scheduler
        scheduler.step()

        # Đánh giá trên tập test
        test_loss, test_acc, test_f1 = evaluate(model, test_loader, criterion=criterion, device=device)
        epoch_time = time.time() - epoch_start

        row_metric = {
            "seed": seed,
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

        print(f"Seed {seed} | Epoch [{epoch+1:02d}/{epochs:02d}] "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
              f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}% | "
              f"Time: {epoch_time:.1f}s")

        # Lưu checkpoint latest cuối mỗi epoch để resume
        ckpt_state = {
            "epoch": epoch + 1,
            "seed": seed,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "random_states": get_random_states(),
            "history": history,
            "best_acc": max(best_acc, test_acc)
        }
        save_checkpoint(ckpt_state, save_dir, filename=f"warmup_seed{seed}_latest.pth")

        # Cập nhật warmup_metrics.csv
        df = pd.DataFrame(history)
        if os.path.exists(metrics_file):
            existing_df = pd.read_csv(metrics_file)
            # Lọc bỏ các dòng cũ của seed này để ghi đè sạch
            filtered_df = existing_df[~((existing_df["seed"] == seed) & (existing_df["epoch"] <= epoch + 1))]
            combined_df = pd.concat([filtered_df, df]).sort_values(by=["seed", "epoch"])
            combined_df.to_csv(metrics_file, index=False)
        else:
            df.to_csv(metrics_file, index=False)

    # Khi hoàn thành đủ 20 epoch: Lưu checkpoint chính thức
    final_pth = os.path.join(save_dir, f"warmup_seed{seed}.pth")
    torch.save({
        "epoch": epochs,
        "seed": seed,
        "model_state_dict": model.state_dict(),
        "final_test_acc": test_acc,
        "final_macro_f1": test_f1
    }, final_pth)
    print(f"\n✅ Đã lưu checkpoint hoàn chỉnh: {final_pth}")

def main():
    parser = argparse.ArgumentParser(description="CIFAR-10 ResNet-18 Warmup")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44], help="Danh sách random seed")
    parser.add_argument("--epochs", type=int, default=20, help="Số epoch warm-up (mặc định 20)")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--data_dir", type=str, default="./data", help="Đường dẫn dữ liệu")
    parser.add_argument("--save_dir", type=str, default="./results/warmup", help="Thư mục lưu weights")
    args = parser.parse_args()

    for s in args.seeds:
        train_warmup_one_seed(
            seed=s,
            epochs=args.epochs,
            batch_size=args.batch_size,
            data_dir=args.data_dir,
            save_dir=args.save_dir
        )

if __name__ == "__main__":
    main()
