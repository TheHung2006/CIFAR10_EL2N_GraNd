import os
import time
import argparse
import numpy as np
import pandas as pd
from collections import Counter
import torchvision
import torchvision.transforms as transforms

CIFAR10_CLASSES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]

def load_labels(data_dir="./data"):
    """Lấy nhãn thực tế của 50.000 mẫu train CIFAR-10."""
    train_set = torchvision.datasets.CIFAR10(
        root=data_dir, train=True, download=True,
        transform=transforms.ToTensor()
    )
    return np.array(train_set.targets)

def select_coreset_topk(scores, k):
    """
    Xếp score giảm dần, chọn Top-k toàn cục:
    Trả về: danh sách các index mẫu được chọn (k phần tử).
    """
    sorted_indices = np.argsort(scores)[::-1]
    top_indices = sorted_indices[:k]
    return top_indices

def run_coreset_selection(
    scores_dir="./results/scores",
    save_dir="./results/coresets",
    data_dir="./data",
    sizes=[500, 1500, 2500, 5000],
    seed=42
):
    os.makedirs(save_dir, exist_ok=True)
    print("==========================================")
    print("BẮT ĐẦU CHỌN CORESET (BƯỚC 4)")
    print(f"Các mốc kích thước coreset: {sizes}")
    print("==========================================")

    # 1. Nạp nhãn CIFAR-10
    labels = load_labels(data_dir=data_dir)
    total_samples = len(labels)

    # 2. Nạp điểm EL2N và GraNd
    el2n_path = os.path.join(scores_dir, "el2n_scores.npy")
    grand_path = os.path.join(scores_dir, "grand_scores.npy")

    if not os.path.exists(el2n_path) or not os.path.exists(grand_path):
        raise FileNotFoundError(f"Chưa đủ file điểm: {el2n_path} hoặc {grand_path}")

    el2n_scores = np.load(el2n_path)
    grand_scores = np.load(grand_path)

    methods = {
        "EL2N": el2n_scores,
        "GraNd": grand_scores
    }

    distribution_rows = []
    selection_times = []
    coreset_dict = {}

    # 3. Tạo các coreset theo EL2N và GraNd
    for method_name, scores in methods.items():
        print(f"\n--> Đang chọn Coreset cho phương pháp: {method_name}")
        for k in sizes:
            t0 = time.time()
            ratio = (k / total_samples) * 100.0

            # Chọn Top-k index
            selected_indices = select_coreset_topk(scores, k)
            sel_time = time.time() - t0

            key = f"{method_name.lower()}_{k}"
            coreset_dict[key] = selected_indices

            # Lưu file .npy
            npy_file = os.path.join(save_dir, f"{method_name.lower()}_coreset_{k}.npy")
            np.save(npy_file, selected_indices)

            # Lưu file .csv (index, label, score)
            csv_file = os.path.join(save_dir, f"{method_name.lower()}_coreset_{k}.csv")
            df_coreset = pd.DataFrame({
                "index": selected_indices,
                "label": labels[selected_indices],
                "score": scores[selected_indices]
            })
            df_coreset.to_csv(csv_file, index=False)

            # Thống kê phân bố lớp
            class_counts = Counter(labels[selected_indices])
            dist_row = {
                "method": method_name,
                "size": k,
                "ratio_pct": ratio,
            }
            for cls_idx, cls_name in enumerate(CIFAR10_CLASSES):
                dist_row[cls_name] = class_counts.get(cls_idx, 0)

            # Độ lệch chuẩn giữa các lớp (đo mức độ lệch phân bố)
            counts_list = [dist_row[cls_name] for cls_name in CIFAR10_CLASSES]
            dist_row["std_class_dist"] = round(float(np.std(counts_list)), 2)

            distribution_rows.append(dist_row)
            selection_times.append({
                "method": method_name,
                "size": k,
                "time_sec": round(sel_time, 4)
            })

            print(f"  ✓ {method_name} - {k} ảnh ({ratio:.1f}%): Lưu {npy_file} | Std phân bố lớp: {dist_row['std_class_dist']}")

    # 4. Tạo thêm Baseline Random Coreset (để đối chứng)
    print(f"\n--> Đang tạo Random Coreset (Baseline)...")
    np.random.seed(seed)
    perm = np.random.permutation(total_samples)
    for k in sizes:
        rand_indices = perm[:k]
        np.save(os.path.join(save_dir, f"random_coreset_{k}.npy"), rand_indices)

        class_counts = Counter(labels[rand_indices])
        dist_row = {
            "method": "Random",
            "size": k,
            "ratio_pct": (k / total_samples) * 100.0,
        }
        for cls_idx, cls_name in enumerate(CIFAR10_CLASSES):
            dist_row[cls_name] = class_counts.get(cls_idx, 0)
        counts_list = [dist_row[cls_name] for cls_name in CIFAR10_CLASSES]
        dist_row["std_class_dist"] = round(float(np.std(counts_list)), 2)
        distribution_rows.append(dist_row)

    # 5. Lưu bảng phân bố lớp và thời gian chọn
    dist_df = pd.DataFrame(distribution_rows)
    dist_csv = os.path.join(save_dir, "coreset_distribution.csv")
    dist_df.to_csv(dist_csv, index=False)
    print(f"\n✓ Đã lưu bảng phân bố lớp: {dist_csv}")

    time_df = pd.DataFrame(selection_times)
    time_csv = os.path.join(save_dir, "coreset_selection_time.csv")
    time_df.to_csv(time_csv, index=False)
    print(f"✓ Đã lưu thời gian chọn coreset: {time_csv}")

    # 6. Đo độ trùng lặp (Overlap Jaccard) giữa EL2N và GraNd
    print(f"\n--- ĐỘ TRÙNG LẶP CORESET (EL2N vs GraNd) ---")
    for k in sizes:
        set_el2n = set(coreset_dict[f"el2n_{k}"])
        set_grand = set(coreset_dict[f"grand_{k}"])
        intersection = len(set_el2n.intersection(set_grand))
        overlap_pct = (intersection / k) * 100.0
        print(f"Size {k:4d} ảnh: Trùng nhau {intersection:4d}/{k} ảnh ({overlap_pct:.2f}%)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores_dir", type=str, default="./results/scores")
    parser.add_argument("--save_dir", type=str, default="./results/coresets")
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--sizes", type=int, nargs="+", default=[500, 1500, 2500, 5000])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    run_coreset_selection(
        scores_dir=args.scores_dir,
        save_dir=args.save_dir,
        data_dir=args.data_dir,
        sizes=args.sizes,
        seed=args.seed
    )
