import torch


def get_device():
    """Apple GPU (mps) on a Mac, NVIDIA GPU (cuda) if there is one, otherwise cpu."""
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"
