"""Scores of every ResNet expert on every image the combiners use, saved with data.features.

One feature set "<arch>_experts" with one column per expert (in EXPERTS order) holding its raw logit
(> 0 = fake). load_scores() turns them into calibrated probabilities sigmoid(logit / T). Every image is
read once and passed through all experts. Held-out experts are not removed here; the combiners drop
them per fold. Only images without scores yet are computed, so a stopped run continues where it left off
and images added to the manifest later (e.g. a seen-test pool) are picked up by running it again.
"""
import json
import time

import numpy as np
import torch

from common.device import get_device
from common.log import Progress, format_duration, log
from data import config
from data.features import INFO_FILE, FeatureStore, load_features
from data.images import image_loader, load_manifest, select
from resnet.experts import EXPERTS
from resnet.model import ARCH, EVAL_PREPROCESS, EVAL_TRANSFORM, checkpoint_id, load_checkpoint, load_expert
from resnet.train import to_device

BATCH_SIZE = 128
NUM_WORKERS = 4
LOG_EVERY = 50  # batches
# expert_train is left out: each expert has seen those images, so their scores would be in-sample.
SCORE_POOLS = None  # None = every pool in the manifest except expert_train


def feature_name(arch=ARCH, limit=None):
    return f"{arch}_experts" if limit is None else f"{arch}_experts_limit{limit}"


def score_pools(manifest):
    return SCORE_POOLS or sorted(set(manifest.pool) - {"expert_train"})


def store_info(arch=ARCH, limit=None):
    """What produced the scores. Retrained experts have other checkpoint ids and temperatures, so their
    scores can't end up in the same store as the old ones (FeatureStore raises)."""
    infos = {expert: load_checkpoint(expert, arch, limit)["info"] for expert in EXPERTS}
    return {
        "model": arch,
        "output": "raw logit per expert (> 0 = fake), one column per expert in the order of 'experts'",
        "preprocess": EVAL_PREPROCESS,
        "experts": EXPERTS,
        "temperatures": {expert: info["temperature"] for expert, info in infos.items()},
        "checkpoints": {expert: checkpoint_id(expert, arch, limit) for expert in EXPERTS},
    }


def score(batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, limit=None, arch=ARCH):
    """Score every image of the scored pools that has no scores yet.

    limit: use the experts of a test run (resnet.train with the same limit) and score at most
    this many images per pool, generator and label.
    """
    store = FeatureStore(feature_name(arch, limit), store_info(arch, limit))
    manifest = load_manifest()
    rows = select(manifest, pool=score_pools(manifest))
    if limit is not None:
        rows = rows.groupby(["pool", "generator", "label"]).head(limit)
    done = store.done_files()
    todo = [file for file in rows.file if file not in done]
    log(f"{store.dir.name}: {len(rows)} images in pools {score_pools(manifest)}, "
        f"{len(rows) - len(todo)} done, {len(todo)} to do")
    if not todo:
        return

    device = get_device()
    experts = [load_expert(expert, device, arch, limit)[0] for expert in EXPERTS]
    log(f"{len(experts)} {arch} experts on {device}, batch size {batch_size}, {num_workers} workers")

    start = time.time()
    progress = Progress(len(todo))
    with torch.no_grad():
        for batch, (images, indices) in enumerate(image_loader(todo, EVAL_TRANSFORM, batch_size, num_workers), 1):
            images = to_device(images, device)
            with torch.autocast("cuda", enabled=device == "cuda"):
                logits = torch.stack([model(images).float().squeeze(1) for model in experts], dim=1)
            store.add([todo[i] for i in indices.tolist()], logits.cpu().numpy())
            progress.step(len(indices))
            if batch % LOG_EVERY == 0:
                finish = time.strftime("%H:%M", time.localtime(time.time() + progress.remaining_seconds()))
                log(f"{progress}, done at ~{finish}")

    store.flush()
    log(f"{store.dir.name}: {len(todo)} images in {format_duration(time.time() - start)}")


def check(limit=None, arch=ARCH):
    """Log whether every image of the scored pools has scores. Returns the rows without scores."""
    manifest = load_manifest()
    rows = select(manifest, pool=score_pools(manifest))
    if limit is not None:
        rows = rows.groupby(["pool", "generator", "label"]).head(limit)
    store = FeatureStore(feature_name(arch, limit))
    missing = rows[~rows.file.isin(store.done_files())]
    log(f"{store.dir.name}: {len(rows) - len(missing)}/{len(rows)} images of pools {score_pools(manifest)} "
        f"have scores, {len(store.parts())} parts")
    if len(missing):
        counts = missing.groupby(["pool", "generator", "label"]).size()
        for (pool, generator, label), count in counts.items():
            log(f"  MISSING {pool:12s} {generator:24s} {config.LABEL_NAMES[label]:4s} {count}")
    else:
        log("all good")
    return missing


def load_scores(pool=None, generator=None, label=None, calibrated=True, limit=None, arch=ARCH):
    """Expert scores of the manifest rows matching the filters (see data.images.select).

    Returns (X, rows) like data.features.load_features: X has one column per expert in EXPERTS order,
    calibrated probabilities sigmoid(logit / T) or, with calibrated=False, raw logits.
    """
    name = feature_name(arch, limit)
    X, rows = load_features(name, pool=pool, generator=generator, label=label)
    if calibrated:
        temperatures = json.loads((FeatureStore(name).dir / INFO_FILE).read_text())["temperatures"]
        X = 1 / (1 + np.exp(-X / np.array([temperatures[expert] for expert in EXPERTS], dtype=np.float32)))
    return X, rows


if __name__ == "__main__":
    score()
