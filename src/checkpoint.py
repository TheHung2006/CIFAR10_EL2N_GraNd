import os
import random
import numpy as np
import torch

def set_seed(seed=42):
    """Cố định hạt giống ngẫu nhiên đảm bảo tính tái lập (reproducibility)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def get_random_states():
    """Lấy trạng thái ngẫu nhiên hiện tại để lưu vào checkpoint."""
    states = {
        "python_random": random.getstate(),
        "numpy_random": np.random.get_state(),
        "torch_random": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        states["torch_cuda_random"] = torch.cuda.get_rng_state_all()
    return states

def set_random_states(states):
    """Khôi phục trạng thái ngẫu nhiên từ checkpoint."""
    if "python_random" in states:
        random.setstate(states["python_random"])
    if "numpy_random" in states:
        np.random.set_state(states["numpy_random"])
    if "torch_random" in states:
        torch.set_rng_state(states["torch_random"])
    if "torch_cuda_random" in states and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(states["torch_cuda_random"])

def save_checkpoint(state, save_dir, filename="latest.pth"):
    """Lưu checkpoint đầy đủ trạng thái huấn luyện."""
    os.makedirs(save_dir, exist_ok=True)
    filepath = os.path.join(save_dir, filename)
    torch.save(state, filepath)
    return filepath

def load_checkpoint(filepath, model, optimizer=None, scheduler=None, device="cpu"):
    """
    Nạp lại trạng thái huấn luyện từ checkpoint:
    Trả về: (start_epoch, history, best_acc)
    """
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"Không tìm thấy checkpoint: {filepath}")

    checkpoint = torch.load(filepath, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    if "random_states" in checkpoint:
        set_random_states(checkpoint["random_states"])

    start_epoch = checkpoint.get("epoch", 0)
    history = checkpoint.get("history", {})
    best_acc = checkpoint.get("best_acc", 0.0)

    print(f"[Resume] Đã nạp checkpoint thành công từ epoch {start_epoch}")
    return start_epoch, history, best_acc
