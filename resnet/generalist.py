"""One ResNet per leave-one-out fold instead of experts: the ResNet counterpart of meta.generalist (RQ1 baseline).

Each fold's generalist is trained like an expert (resnet.train), but on the expert_train images of every seen
expert together, with their expert_val images for early stopping and the temperature. So the only difference to
the experts is one network instead of several. There is one generalist per fold of meta.loo.folds, with and
without merge_siblings: 7 single folds plus SD+wukong.

Every image of SCORE_POOLS is then scored with every generalist, saved as the feature set "<arch>_generalists"
with one raw logit per fold (FOLDS order), and evaluated like meta.generalist.run_generalist.
Finished generalists and scored images are saved and skipped on a re-run.
"""
import time

import pandas as pd
import torch

from common.device import get_device
from common.log import Progress, format_duration, log
from data import config
from data.features import FeatureStore
from data.images import image_loader, load_manifest, select
from meta.loo import best_threshold, fold_metrics, folds, to_table
from resnet.experts import EXPERTS, check_no_leaks, expert_rows
from resnet.model import EVAL_PREPROCESS, EVAL_TRANSFORM, MODELS_DIR, build_model
from resnet.train import BATCH_SIZE, EPOCHS, NUM_WORKERS, to_device, train_model

ARCH = "resnet18"  # ~2.5x faster than resnet50, and the two expert pools barely differed
# fold name -> held-out experts, the folds of both merge_siblings settings
FOLDS = {fold: experts for merge_siblings in (False, True) for fold, experts in folds(merge_siblings)}
# meta_val sets the threshold, test is evaluated. The generalists aren't combined, so meta_train isn't needed.
SCORE_POOLS = ["meta_val", "test"]
LOG_EVERY = 50  # batches


def generalists_dir(arch=ARCH, limit=None):
    """Folder with one checkpoint per fold, next to the experts. Test runs (limit) get their own folder."""
    name = f"{arch}_generalists" if limit is None else f"{arch}_generalists_limit{limit}"
    return config.dataset_dir() / MODELS_DIR / name


def checkpoint_path(fold, arch=ARCH, limit=None):
    return generalists_dir(arch, limit) / f"{fold}.pt"


def load_generalist(fold, device, arch=ARCH, limit=None):
    """Returns (model in eval mode on device, info)."""
    checkpoint = torch.load(checkpoint_path(fold, arch, limit), map_location="cpu")
    model = build_model(arch, pretrained=False)
    model.load_state_dict(checkpoint["state_dict"])
    return model.to(device).eval(), checkpoint["info"]


def seen_rows(manifest, held_out, pool, limit=None):
    """The images of pool that the experts not in held_out train on, all together."""
    rows = pd.concat([expert_rows(manifest, expert, pool, limit) for expert in EXPERTS if expert not in held_out])
    return rows.drop_duplicates("file").sort_values("file")


def train_generalist(fold, manifest, device, epochs=EPOCHS, limit=None,
                     batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, arch=ARCH, queued=0):
    """Train, calibrate and save the generalist of one fold. Returns its info. Skipped if already trained.

    limit: at most this many images per label, pool and seen expert (for test runs, saved separately).
    queued: networks still to train after this one, for the time estimate of the whole run.
    """
    path = checkpoint_path(fold, arch, limit)
    if path.exists():
        log(f"[{fold}] already trained, skipping (delete {path} to retrain)")
        return torch.load(path, map_location="cpu")["info"]
    held_out = FOLDS[fold]
    experts = [expert for expert in EXPERTS if expert not in held_out]
    info = {
        "fold": fold,
        "held_out": held_out,
        "experts": experts,
        "generators": [generator for expert in experts for generator in EXPERTS[expert]],
    }
    return train_model(fold, seen_rows(manifest, held_out, "expert_train", limit),
                       seen_rows(manifest, held_out, "expert_val", limit),
                       path, info, device, epochs, limit, batch_size, num_workers, arch, queued)


def train_all(fold_names=None, epochs=EPOCHS, limit=None, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, arch=ARCH):
    """Train every generalist that isn't trained yet. Returns {fold: info}."""
    manifest = load_manifest()
    check_no_leaks(manifest)
    device = get_device()
    fold_names = fold_names or list(FOLDS)
    untrained = [fold for fold in fold_names if not checkpoint_path(fold, arch, limit).exists()]
    infos = {}
    for fold in fold_names:
        queued = len(untrained) - untrained.index(fold) - 1 if fold in untrained else 0
        infos[fold] = train_generalist(fold, manifest, device, epochs, limit, batch_size, num_workers, arch, queued)
        if device == "mps":
            torch.mps.empty_cache()
        elif device == "cuda":
            torch.cuda.empty_cache()
    return infos


def feature_name(arch=ARCH, limit=None):
    return f"{arch}_generalists" if limit is None else f"{arch}_generalists_limit{limit}"


def scored_rows(limit=None):
    rows = select(load_manifest(), pool=SCORE_POOLS)
    return rows if limit is None else rows.groupby(["pool", "generator", "label"]).head(limit)


def score(batch_size=BATCH_SIZE * 2, num_workers=NUM_WORKERS, limit=None, arch=ARCH):
    """Score every image of SCORE_POOLS that has no scores yet with every generalist."""
    device = get_device()
    loaded = {fold: load_generalist(fold, device, arch, limit) for fold in FOLDS}
    info = {
        "model": arch,
        "output": "raw logit per generalist (> 0 = fake), one column per fold in the order of 'folds'",
        "preprocess": EVAL_PREPROCESS,
        "folds": FOLDS,
        "temperatures": {fold: fold_info["temperature"] for fold, (_, fold_info) in loaded.items()},
    }
    store = FeatureStore(feature_name(arch, limit), info)
    rows = scored_rows(limit)
    done = store.done_files()
    todo = [file for file in rows.file if file not in done]
    log(f"{store.dir.name}: {len(rows)} images in pools {SCORE_POOLS}, {len(rows) - len(todo)} done, {len(todo)} to do")
    if not todo:
        return

    models = [model for model, _ in loaded.values()]
    start = time.time()
    progress = Progress(len(todo))
    with torch.no_grad():
        for batch, (images, indices) in enumerate(image_loader(todo, EVAL_TRANSFORM, batch_size, num_workers), 1):
            images = to_device(images, device)
            with torch.autocast("cuda", enabled=device == "cuda"):
                logits = torch.stack([model(images).float().squeeze(1) for model in models], dim=1)
            store.add([todo[i] for i in indices.tolist()], logits.cpu().numpy())
            progress.step(len(indices))
            if batch % LOG_EVERY == 0:
                log(f"{progress}")
    store.flush()
    log(f"{store.dir.name}: {len(todo)} images in {format_duration(time.time() - start)}")


def run_generalist(merge_siblings=False, eval_pool="test", limit=None, arch=ARCH, scores=None):
    """One row per fold plus the mean, with the same columns as meta.generalist.run_generalist.

    scores: optional dict, filled with {fold: (labels, scores)} of the held-out images, as in meta.loo.run_loo.
    """
    store = FeatureStore(feature_name(arch, limit))
    rows = scored_rows(limit)
    rows_val, rows_eval = rows[rows.pool == "meta_val"], rows[rows.pool == eval_pool]
    X_val, X_eval = store.load(rows_val.file), store.load(rows_eval.file)
    y_eval = rows_eval.label.to_numpy()
    results = []
    for fold, experts in folds(merge_siblings):
        column = list(FOLDS).index(fold)
        generators = [generator for expert in experts for generator in EXPERTS[expert]]
        is_val = ~rows_val.generator.isin(generators).to_numpy()
        is_unseen = rows_eval.generator.isin(generators).to_numpy()
        threshold = best_threshold(X_val[is_val, column], rows_val.label.to_numpy()[is_val])
        if scores is not None:
            scores[fold] = (y_eval[is_unseen].astype("int8"), X_eval[is_unseen, column].astype("float32"))
        path = checkpoint_path(fold, arch, limit)
        # The scores are enough to evaluate; without the checkpoint, count the training images instead.
        n_train = (torch.load(path, map_location="cpu")["info"]["n_train"] if path.exists()
                   else len(seen_rows(load_manifest(), experts, "expert_train", limit)))
        results.append({
            "held out": fold,
            "generators": ", ".join(generators),
            "n_train": n_train,
            "n_val": int(is_val.sum()),
            "n_eval": int(is_unseen.sum()),
            **fold_metrics(X_eval[is_unseen, column], y_eval[is_unseen],
                           X_eval[~is_unseen, column], y_eval[~is_unseen], threshold),
        })
    return to_table(results)


if __name__ == "__main__":
    train_all()
    score()
