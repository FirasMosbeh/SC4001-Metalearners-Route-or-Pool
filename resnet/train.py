"""Train one ResNet per expert (real vs. that expert's fakes) and calibrate it.

Each expert fine-tunes the whole ImageNet-pretrained network on its own expert_train images only,
checks expert_val after every epoch, keeps the epoch with the lowest validation loss (accuracy and
AUROC hit ~100% quickly and stop being informative) and stops after PATIENCE epochs without
improvement. Then one temperature per expert is fitted on expert_val (temperature scaling).
meta_train, meta_val and test images are never touched here.

Finished experts are saved and skipped on a re-run, so a stopped run continues with the next expert.
"""
import time

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import minimize_scalar
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from common.device import get_device
from common.log import format_duration, log
from data import config
from data.images import ImageDataset, load_manifest
from resnet.experts import EXPERTS, check_no_leaks, expert_rows
from resnet.model import ARCH, EVAL_TRANSFORM, TRAIN_TRANSFORM, build_model, checkpoint_path, load_checkpoint

EPOCHS = 8  # at most; early stopping usually ends earlier
PATIENCE = 2  # epochs without a lower validation loss before stopping
BATCH_SIZE = 64  # fits ResNet-50 training on a 16 GB Mac
LR = 1e-4
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 4
LOG_EVERY = 50  # batches
# Perfectly separated validation images would push the temperature towards 0 (ever sharper). A sharper
# expert is more overconfident on generators it has never seen, which is exactly our test case, so
# temperatures may only soften (T >= 1), never sharpen. Raw logits are saved too, for recalibration.
TEMPERATURE_BOUNDS = (1.0, 50.0)


def make_loader(files, transform, batch_size, num_workers, shuffle=False):
    """Batches of (images, indices into files). Shuffled loaders drop a last partial batch (BatchNorm)."""
    return DataLoader(
        ImageDataset(files, transform), batch_size=batch_size, shuffle=shuffle,
        drop_last=shuffle and len(files) > batch_size, num_workers=num_workers,
        persistent_workers=num_workers > 0,  # macOS workers are slow to start, so keep them between epochs
        generator=torch.Generator().manual_seed(config.SEED) if shuffle else None,
    )


def to_device(images, device):
    images = images.to(device, non_blocking=True)
    # channels_last speeds up convolutions on NVIDIA GPUs; on mps it doesn't help.
    return images.to(memory_format=torch.channels_last) if device == "cuda" else images


@torch.no_grad()
def predict_logits(model, loader, device):
    """Logits of all images of a non-shuffled loader, in the order of its files."""
    model.eval()
    logits = np.empty(len(loader.dataset), dtype=np.float32)
    for images, indices in loader:
        with torch.autocast("cuda", enabled=device == "cuda"):
            logits[indices.numpy()] = model(to_device(images, device)).float().squeeze(1).cpu().numpy()
    return logits


def bce(logits, labels):
    """Mean binary cross-entropy of sigmoid(logits), numerically stable."""
    return float(np.mean(np.logaddexp(0, logits) - labels * logits))


def metrics(logits, labels):
    return {
        "loss": bce(logits, labels),
        "acc": float(((logits > 0) == (labels == 1)).mean()),
        "auc": float(roc_auc_score(labels, logits)),
    }


def fit_temperature(logits, labels, bounds=TEMPERATURE_BOUNDS):
    """Temperature T within bounds that minimises the loss of sigmoid(logits / T)."""
    result = minimize_scalar(lambda log_t: bce(logits / np.exp(log_t), labels),
                             bounds=np.log(bounds), method="bounded")
    return float(np.exp(result.x))


def train_epoch(model, loader, labels, optimizer, scheduler, scaler, device, epoch):
    """One pass over the training images. Returns the mean training loss."""
    model.train()
    total_loss, seen, start = 0.0, 0, time.time()
    for batch, (images, indices) in enumerate(loader, 1):
        targets = labels[indices].to(device, non_blocking=True)
        with torch.autocast("cuda", enabled=device == "cuda"):  # mixed precision gave no speed-up on mps
            loss = F.binary_cross_entropy_with_logits(model(to_device(images, device)).squeeze(1).float(), targets)
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        total_loss += loss.item() * len(indices)
        seen += len(indices)
        if batch % LOG_EVERY == 0:
            log(f"  epoch {epoch} batch {batch}/{len(loader)}: train loss {total_loss / seen:.4f}, "
                f"{seen / (time.time() - start):.0f} images/s")
    return total_loss / seen


def train_expert(expert, manifest, device, epochs=EPOCHS, limit=None,
                 batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, arch=ARCH):
    """Train, calibrate and save one expert. Returns its info. Skipped if the checkpoint already exists.

    limit: at most this many images per label and pool (for test runs, saved in a separate folder).
    """
    path = checkpoint_path(expert, arch, limit)
    if path.exists():
        log(f"[{expert}] already trained, skipping (delete {path} to retrain)")
        return load_checkpoint(expert, arch, limit)["info"]

    train_rows = expert_rows(manifest, expert, "expert_train", limit)
    val_rows = expert_rows(manifest, expert, "expert_val", limit)
    for name, rows in (("expert_train", train_rows), ("expert_val", val_rows)):
        if rows.label.nunique() < 2:
            raise ValueError(f"[{expert}] {name} needs real and fake images, has {rows.label.value_counts().to_dict()}")
    train_labels = torch.tensor(train_rows.label.to_numpy(), dtype=torch.float32)
    val_labels = val_rows.label.to_numpy()

    torch.manual_seed(config.SEED)
    train_loader = make_loader(train_rows.file, TRAIN_TRANSFORM, batch_size, num_workers, shuffle=True)
    val_loader = make_loader(val_rows.file, EVAL_TRANSFORM, batch_size * 2, num_workers)

    model = build_model(arch).to(device)
    if device == "cuda":
        model = model.to(memory_format=torch.channels_last)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=LR, total_steps=epochs * len(train_loader),
                                                    pct_start=0.1)
    scaler = torch.amp.GradScaler("cuda", enabled=device == "cuda")

    log(f"[{expert}] {arch} on {device}: {len(train_rows)} train images ({train_rows.label.mean():.0%} fake), "
        f"{len(val_rows)} val images, generators {EXPERTS[expert]}")
    best_loss, best_state, best_epoch, history = float("inf"), None, 0, []
    start = time.time()
    for epoch in range(1, epochs + 1):
        train_loss = train_epoch(model, train_loader, train_labels, optimizer, scheduler, scaler, device, epoch)
        val = metrics(predict_logits(model, val_loader, device), val_labels)
        history.append({"epoch": epoch, "train_loss": train_loss, **{f"val_{k}": v for k, v in val.items()}})
        log(f"[{expert}] epoch {epoch}: train loss {train_loss:.4f} | val loss {val['loss']:.4f} "
            f"acc {val['acc']:.4f} auc {val['auc']:.4f} | {format_duration(time.time() - start)}")
        if val["loss"] < best_loss - 1e-5:
            best_loss, best_epoch = val["loss"], epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        elif epoch - best_epoch >= PATIENCE:
            log(f"[{expert}] early stop, best epoch {best_epoch}")
            break

    model.load_state_dict(best_state)
    val_logits = predict_logits(model, val_loader, device)
    temperature = fit_temperature(val_logits, val_labels)
    info = {
        "expert": expert,
        "generators": EXPERTS[expert],
        "arch": arch,
        "temperature": temperature,
        "best_epoch": best_epoch,
        "epochs_run": len(history),
        "val": metrics(val_logits, val_labels),
        "val_calibrated_loss": bce(val_logits / temperature, val_labels),
        "n_train": len(train_rows),
        "n_val": len(val_rows),
        "limit": limit,
        "train_seconds": round(time.time() - start),
        "history": history,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    # Write to a temporary file first, so a crash mid-write can't leave a broken checkpoint that gets skipped.
    temporary = path.with_suffix(".tmp")
    torch.save({"state_dict": best_state, "info": info}, temporary)
    temporary.replace(path)
    log(f"[{expert}] saved: val auc {info['val']['auc']:.4f}, loss {info['val']['loss']:.4f}, "
        f"T={temperature:.3f}, {format_duration(info['train_seconds'])}")
    return info


def train_all(experts=None, epochs=EPOCHS, limit=None, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, arch=ARCH):
    """Train every expert that isn't trained yet. Returns {expert: info}."""
    manifest = load_manifest()
    check_no_leaks(manifest)
    device = get_device()
    infos = {}
    for expert in experts or EXPERTS:
        infos[expert] = train_expert(expert, manifest, device, epochs, limit, batch_size, num_workers, arch)
        if device == "mps":
            torch.mps.empty_cache()
        elif device == "cuda":
            torch.cuda.empty_cache()
    return infos


if __name__ == "__main__":
    train_all()
