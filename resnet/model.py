"""ResNet expert network, its preprocessing and where trained experts are saved."""
import hashlib

import torch
import torch.nn as nn
from torchvision import models, transforms

from data import config

ARCH = "resnet50"  # "resnet18" is ~2.5x faster (M4: 108 vs 42 images/s training)
IMG_SIZE = 224
MODELS_DIR = "models"  # inside the dataset folder

# ImageNet mean/std, because the backbone was pretrained on ImageNet.
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
WEIGHTS = {
    "resnet18": models.ResNet18_Weights.IMAGENET1K_V1,
    "resnet50": models.ResNet50_Weights.IMAGENET1K_V2,
}

# Our images are 256x256, so 224 crops never resize: resizing would blur the pixel traces the expert should learn.
TRAIN_TRANSFORM = transforms.Compose([
    transforms.RandomCrop(IMG_SIZE),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])
EVAL_TRANSFORM = transforms.Compose([
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])
EVAL_PREPROCESS = f"CenterCrop({IMG_SIZE}), ToTensor, Normalize(ImageNet mean/std)"


def build_model(arch=ARCH, pretrained=True):
    """ResNet with the 1000-class head replaced by one logit (> 0 = fake)."""
    model = getattr(models, arch)(weights=WEIGHTS[arch] if pretrained else None)
    model.fc = nn.Linear(model.fc.in_features, 1)
    return model


def experts_dir(arch=ARCH, limit=None):
    """Folder with one checkpoint per expert. Test runs (limit) get their own folder."""
    name = arch if limit is None else f"{arch}_limit{limit}"
    return config.dataset_dir() / MODELS_DIR / name


def checkpoint_path(expert, arch=ARCH, limit=None):
    return experts_dir(arch, limit) / f"{expert}.pt"


def load_checkpoint(expert, arch=ARCH, limit=None):
    """{"state_dict", "info"}, info as saved by resnet.train."""
    return torch.load(checkpoint_path(expert, arch, limit), map_location="cpu")


def load_expert(expert, device, arch=ARCH, limit=None):
    """Returns (model in eval mode on device, info)."""
    checkpoint = load_checkpoint(expert, arch, limit)
    model = build_model(arch, pretrained=False)
    model.load_state_dict(checkpoint["state_dict"])
    return model.to(device).eval(), checkpoint["info"]


def checkpoint_id(expert, arch=ARCH, limit=None):
    """Short hash of the checkpoint file, so scores can be traced to the exact weights."""
    return hashlib.md5(checkpoint_path(expert, arch, limit).read_bytes()).hexdigest()[:12]
