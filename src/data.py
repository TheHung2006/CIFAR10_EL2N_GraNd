import os
import json
import torch
from torch.utils.data import DataLoader, Subset, Dataset
import torchvision
import torchvision.transforms as transforms
from collections import Counter

# Chuẩn hóa CIFAR-10 chuẩn
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)

class IndexedDataset(Dataset):
    """
    Dataset wrapper trả về thêm index của mẫu: (image, target, index).
    Dùng cho việc tính điểm EL2N và GraNd để đảm bảo ánh xạ chính xác điểm số vào từng mẫu.
    """
    def __init__(self, dataset):
        self.dataset = dataset

    def __getitem__(self, index):
        data, target = self.dataset[index]
        return data, target, index

    def __len__(self):
        return len(self.dataset)

def get_transforms():
    """
    Trả về bộ biến đổi dữ liệu:
    - train_transform: Có Data Augmentation (Random Crop, Random Flip) dùng khi huấn luyện.
    - eval_transform: Không Augmentation (chỉ Normalize) dùng khi tính score và test.
    """
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])

    eval_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])

    return train_transform, eval_transform

def get_train_loader(data_dir="./data", batch_size=128, num_workers=2, subset_indices=None):
    """DataLoader cho huấn luyện (Warm-up hoặc Huấn luyện trên Coreset)."""
    train_transform, _ = get_transforms()
    train_set = torchvision.datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=train_transform
    )
    if subset_indices is not None:
        train_set = Subset(train_set, subset_indices)

    return DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=torch.cuda.is_available()
    )

def get_scoring_loader(data_dir="./data", batch_size=128, num_workers=2):
    """
    DataLoader cho tính điểm EL2N / GraNd:
    - Dùng eval_transform (KHÔNG Data Augmentation ngẫu nhiên).
    - Trả về (images, targets, indices) với thứ tự giữ nguyên (shuffle=False).
    """
    _, eval_transform = get_transforms()
    raw_train_set = torchvision.datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=eval_transform
    )
    indexed_set = IndexedDataset(raw_train_set)

    return DataLoader(
        indexed_set, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=torch.cuda.is_available()
    )

def get_test_loader(data_dir="./data", batch_size=128, num_workers=2):
    """DataLoader cho đánh giá tập Test."""
    _, eval_transform = get_transforms()
    test_set = torchvision.datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=eval_transform
    )
    return DataLoader(
        test_set, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=torch.cuda.is_available()
    )

def inspect_dataset(data_dir="./data", save_path="data_info.json"):
    """Kiểm tra số lượng ảnh, phân bố các lớp và lưu lại data_info.json theo kế hoạch."""
    _, eval_transform = get_transforms()
    train_set = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=True, transform=eval_transform)
    test_set = torchvision.datasets.CIFAR10(root=data_dir, train=False, download=True, transform=eval_transform)

    train_dist = dict(Counter(train_set.targets))
    test_dist = dict(Counter(test_set.targets))

    info = {
        "dataset": "CIFAR10",
        "classes": train_set.classes,
        "num_classes": len(train_set.classes),
        "train_samples": len(train_set),
        "test_samples": len(test_set),
        "train_class_distribution": {train_set.classes[k]: v for k, v in train_dist.items()},
        "test_class_distribution": {test_set.classes[k]: v for k, v in test_dist.items()},
    }

    if save_path:
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(info, f, indent=2, ensure_ascii=False)
        print(f"[Info] Đã lưu thông tin dataset vào: {save_path}")

    return info

if __name__ == "__main__":
    info = inspect_dataset(data_dir="./data", save_path="data_info.json")
    print(f"Tổng số ảnh train: {info['train_samples']}, ảnh test: {info['test_samples']}")
    print(f"Phân bố lớp train: {info['train_class_distribution']}")
