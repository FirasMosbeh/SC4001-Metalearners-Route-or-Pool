"""Every meta-learner on every expert pool, saved as one CSV with a row per fold and seed.

Learners that depend on a random start run once per seed, the others once. Results go to
dataset_dir/results/, which is what notebooks/meta_learners.ipynb reads.
"""
import time

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from common.log import format_duration, log
from data import config
from meta.generalist import run_generalist
from meta.learners import LEARNERS
from meta.loo import run_loo

RESULTS_DIR = "results"  # inside the dataset folder
BASES = ["clip", "resnet18", "resnet50"]
GENERALIST = "generalist"  # a CLIP probe on the images themselves, so it only has a row for the clip base
FIXED = ["best_expert", "mean", "max", "sorted_lr", "stacking_lr", "pu_stacking_lr"]
SEEDED = ["mlp", "shuffled_mlp", "deep_sets", "gating", "pu_mlp", "pu_deep_sets", "pu_gating"]
SEEDS = [4001, 4002, 4003]


def results_path(eval_pool="test"):
    return config.dataset_dir() / RESULTS_DIR / f"meta_learners_{eval_pool}.csv"


def scores_path(eval_pool="test"):
    """Held-out labels and scores of every run, {(base, learner, seed, merge_siblings, scale, fold): (labels, scores)}."""
    return config.dataset_dir() / RESULTS_DIR / f"meta_learners_{eval_pool}_scores.joblib"


def run_all(eval_pool="test", bases=BASES, seeds=SEEDS, variants=((True, False), (False, False), (True, True))):
    """variants: (merge_siblings, scale) settings to run. Returns the results, saved after each base together
    with the held-out scores of every run (scores_path), which meta.bootstrap compares."""
    torch.set_num_threads(1)  # these networks are tiny, so more threads only add overhead
    path = results_path(eval_pool)
    path.parent.mkdir(parents=True, exist_ok=True)
    parts, all_scores, start = [], {}, time.time()
    for base in bases:
        for merge_siblings, scale in variants:
            runs = [(name, None) for name in FIXED] + [(name, seed) for name in SEEDED for seed in seeds]
            for name, seed in runs:
                learner = LEARNERS[name]() if seed is None else LEARNERS[name](seed=seed)
                scores = {}
                table = run_loo(learner, base, merge_siblings, eval_pool, scale, scores).drop(index="mean").reset_index()
                parts.append(table.assign(base=base, learner=name, seed=seed, merge_siblings=merge_siblings, scale=scale))
                all_scores.update({(base, name, seed, merge_siblings, scale, fold): s for fold, s in scores.items()})
            if base == "clip":
                scores = {}
                table = run_generalist(merge_siblings, eval_pool, scores).drop(index="mean").reset_index()
                parts.append(table.assign(base=base, learner=GENERALIST, seed=None, merge_siblings=merge_siblings, scale=scale))
                all_scores.update({(base, GENERALIST, None, merge_siblings, scale, fold): s for fold, s in scores.items()})
            pd.concat(parts).to_csv(path, index=False)
            joblib.dump(all_scores, scores_path(eval_pool))
            log(f"{base}, merge_siblings={merge_siblings}, scale={scale}: done, {format_duration(time.time() - start)}")
    return pd.concat(parts)


def load_results(eval_pool="test"):
    return pd.read_csv(results_path(eval_pool))


def summary(results, metric="auc", merge_siblings=True, scale=False):
    """Learners (rows) x bases (columns): mean of metric over seeds, then over folds."""
    rows = results[(results.merge_siblings == merge_siblings) & (results.scale == scale)]
    per_fold = rows.groupby(["learner", "base", "held out"])[metric].mean()
    table = per_fold.groupby(["learner", "base"]).mean().unstack("base")
    return table.reindex([name for name in [GENERALIST] + FIXED + SEEDED if name in table.index])[[b for b in BASES if b in table]]


def plot_summary(table, title):
    """Heat map of a summary table, one hue from light (low) to dark (high)."""
    values = table.to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(1.6 * values.shape[1] + 2.4, 0.42 * values.shape[0] + 1.2))
    cmap = plt.matplotlib.colors.LinearSegmentedColormap.from_list(
        "blue", ["#f0efec", "#9ec5f4", "#3987e5", "#1c5cab", "#0d366b"])
    low, high = np.nanmin(values), np.nanmax(values)
    image = ax.imshow(values, cmap=cmap, vmin=low, vmax=high, aspect="auto")
    ax.set_xticks(range(values.shape[1]), table.columns)
    ax.set_yticks(range(values.shape[0]), table.index)
    ax.set_title(title, loc="left")
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            if not np.isnan(values[i, j]):
                ax.text(j, i, f"{values[i, j]:.3f}", ha="center", va="center", fontsize=9,
                        color="white" if values[i, j] > (low + high) / 2 else "#0b0b0b")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    run_all()
