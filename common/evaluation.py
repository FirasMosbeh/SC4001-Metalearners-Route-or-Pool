"""Evaluation of an expert pool's scores, shared by the ResNet and CLIP experts.

Works on any score set saved with data.features that has one raw logit per expert (> 0 = fake), one
column per expert in EXPERTS order: resnet.score.feature_name(arch, limit) or clip.score.SCORES_NAME.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from data import config
from data.features import load_features
from resnet.experts import EXPERTS

SHORT_NAMES = {"stable_diffusion_v_1_4": "SD 1.4", "stable_diffusion_v_1_5": "SD 1.5"}


def cross_auroc(scores_name, pool="meta_val"):
    """AUROC of every expert (rows) on every generator's images of the pool (columns).

    Expect ~1.0 on the diagonal, high values for sibling generators (SD <-> wukong, ADM -> glide/BigGAN)
    and lower ones elsewhere. Everything ~1.0 would point to a shortcut (image size or JPEG traces).
    """
    X, rows = load_features(scores_name, pool=pool)
    table = pd.DataFrame(index=list(EXPERTS), columns=config.GENERATORS, dtype=float)
    for generator in config.GENERATORS:
        is_generator = (rows.generator == generator).to_numpy()
        if rows.label[is_generator].nunique() == 2:
            for column, expert in enumerate(EXPERTS):
                table.loc[expert, generator] = roc_auc_score(rows.label[is_generator], X[is_generator, column])
    return table


def plot_cross_auroc(table, title, pool="meta_val"):
    """Heat map of a cross_auroc table."""
    values = table.to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(7.6, 5.4))
    cmap = plt.matplotlib.colors.LinearSegmentedColormap.from_list(
        "blue", ["#f0efec", "#9ec5f4", "#3987e5", "#1c5cab", "#0d366b"])  # one hue, light -> dark
    image = ax.imshow(values, cmap=cmap, vmin=0.5, vmax=1.0)
    ax.set_xticks(range(len(table.columns)), [SHORT_NAMES.get(g, g) for g in table.columns], rotation=35, ha="right")
    ax.set_yticks(range(len(table.index)), table.index)
    ax.set_xlabel(f"images from generator ({pool})")
    ax.set_ylabel("expert")
    ax.set_title(title, loc="left")
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            if not np.isnan(values[i, j]):
                ax.text(j, i, f"{values[i, j]:.2f}", ha="center", va="center", fontsize=8,
                        color="white" if values[i, j] > 0.8 else "#0b0b0b")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="AUROC (0.5 = chance)")
    plt.tight_layout()
    plt.show()
