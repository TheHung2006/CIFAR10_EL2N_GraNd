import torch
import torch.nn as nn
import torchvision.models as models

class CIFAR10ResNet18(nn.Module):
    """
    ResNet-18 được điều chỉnh chuẩn cho ảnh kích thước nhỏ 32x32 của CIFAR-10:
    - Thay Conv1 (7x7, stride 2) thành (3x3, stride 1, padding 1).
    - Bỏ MaxPool đầu tiên (Identity) để không làm mất thông tin không gian quá sớm.
    - Lớp phân loại cuối cùng chuyển về 10 lớp.
    """
    def __init__(self, num_classes=10):
        super(CIFAR10ResNet18, self).__init__()
        # Khởi tạo ResNet-18 cơ bản từ torchvision
        resnet = models.resnet18(weights=None)

        # Điều chỉnh lớp Conv đầu tiên
        resnet.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        # Bỏ maxpool (thay bằng Identity)
        resnet.maxpool = nn.Identity()
        # Đổi lớp fully connected cuối cùng thành num_classes (10)
        resnet.fc = nn.Linear(resnet.fc.in_features, num_classes)

        self.model = resnet

    def forward(self, x):
        return self.model(x)

def get_model(num_classes=10, device="cpu"):
    model = CIFAR10ResNet18(num_classes=num_classes)
    return model.to(device)

if __name__ == "__main__":
    # Test nhanh kích thước đầu vào và đầu ra
    x = torch.randn(2, 3, 32, 32)
    net = get_model(num_classes=10)
    out = net(x)
    print(f"Kiểm tra ResNet18 CIFAR-10:")
    print(f"Input shape : {x.shape}")
    print(f"Output shape: {out.shape} (Dự kiến: [2, 10])")
    total_params = sum(p.numel() for p in net.parameters())
    print(f"Tổng số tham số: {total_params:,}")
