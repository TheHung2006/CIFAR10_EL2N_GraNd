import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score

def evaluate(model, test_loader, criterion=None, device="cpu"):
    """
    Đánh giá mô hình trên tập test:
    Trả về: (test_loss, test_accuracy, test_macro_f1)
    """
    model.eval()
    if criterion is None:
        criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for images, targets in test_loader:
            images, targets = images.to(device), targets.to(device)
            outputs = model(images)
            loss = criterion(outputs, targets)

            total_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    avg_loss = total_loss / len(test_loader.dataset)
    acc = accuracy_score(all_targets, all_preds) * 100.0
    macro_f1 = f1_score(all_targets, all_preds, average="macro")

    return avg_loss, acc, macro_f1
