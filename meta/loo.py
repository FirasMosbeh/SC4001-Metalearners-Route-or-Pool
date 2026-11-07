"""Leave-one-out evaluation of a meta-learner on the scores of one expert pool (the base).

One fold per expert: its generators are held out. The learner is fitted on the meta_train images of the
other generators, without the held-out expert's column, and evaluated on the images of the held-out
generators in eval_pool, which neither the experts nor the learner have seen. meta_val of the other
generators is passed to fit as validation and sets the accuracy threshold.

eval_pool is "test" for the final numbers. Use "meta_val" while developing, so test stays untouched; its
held-out images are unseen too, but seen_auc is then measured on the images used for early stopping.
"""
import copy
from functools import lru_cache

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from data.features import load_features
from resnet.experts import EXPERTS
from resnet.score import feature_name

# base -> saved expert scores (one raw logit per expert, columns in EXPERTS order)
BASES = {
    "clip": "clip_vit-l14_experts",  # clip.score.SCORES_NAME, which would import transformers
    "resnet18": feature_name("resnet18"),
    "resnet50": feature_name("resnet50"),
}
# These experts detect each other's generators at AUROC ~0.99, so neither is unseen while the other stays.
SIBLINGS = ["SD", "wukong"]
METRICS = ["auc", "ap", "acc", "seen_auc", "gap", "expert_mean_auc", "expert_best_auc"]


def folds(merge_siblings=False):
    """(fold name, held-out experts), one per expert. merge_siblings: hold SIBLINGS out together."""
    for expert in EXPERTS:
        if not merge_siblings or expert not in SIBLINGS:
            yield expert, [expert]
    if merge_siblings:
        yield "+".join(SIBLINGS), SIBLINGS


@lru_cache(maxsize=None)
def load_pool(base, pool):
    return load_features(BASES[base], pool=pool)


def best_threshold(scores, labels):
    """Score threshold with the highest accuracy on these images, searched over their percentiles."""
    candidates = np.quantile(scores, np.linspace(0.01, 0.99, 99))
    return candidates[np.argmax([((scores > t) == (labels == 1)).mean() for t in candidates])]


def fold_metrics(unseen, y_unseen, seen, y_seen, threshold):
    """Scores of one fold's unseen and seen images, as table columns."""
    return {
        "auc": roc_auc_score(y_unseen, unseen),
        "ap": average_precision_score(y_unseen, unseen),
        "acc": float(((unseen > threshold) == (y_unseen == 1)).mean()),
        "seen_auc": roc_auc_score(y_seen, seen),
    }


def to_table(results):
    """One row per fold plus the mean."""
    table = pd.DataFrame(results).set_index("held out")
    table["gap"] = table.seen_auc - table.auc
    metrics = [column for column in METRICS if column in table]
    table.loc["mean", metrics] = table[metrics].mean()
    return table


def own_columns(rows, columns):
    """Per image, the position within columns of the expert of its generator."""
    expert_of = {generator: expert for expert, generators in EXPERTS.items() for generator in generators}
    position = {expert: columns.index(list(EXPERTS).index(expert))
                for expert in EXPERTS if list(EXPERTS).index(expert) in columns}
    return rows.generator.map(expert_of).map(position).to_numpy(dtype=int)


def run_loo(learner, base, merge_siblings=False, eval_pool="test", scale=False):
    """Fit and evaluate a fresh copy of learner per fold. Returns one row per fold plus the mean.

    scale: divide each expert's logits by their standard deviation on the fold's training images (sensitivity
    check: pools differ in logit size, but on meta_val this changed nothing except making gating worse).
    Columns expert_mean_auc and expert_best_auc are the individual experts on the held-out images, averaged
    and at their best; the best one can only be known afterwards, so it's an upper bound.
    """
    X_train, rows_train = load_pool(base, "meta_train")
    X_val, rows_val = load_pool(base, "meta_val")
    X_eval, rows_eval = load_pool(base, eval_pool)
    y_eval = rows_eval.label.to_numpy()
    results = []
    for fold, experts in folds(merge_siblings):
        generators = [generator for expert in experts for generator in EXPERTS[expert]]
        columns = [c for c, expert in enumerate(EXPERTS) if expert not in experts]
        is_train = ~rows_train.generator.isin(generators).to_numpy()
        is_val = ~rows_val.generator.isin(generators).to_numpy()
        is_unseen = rows_eval.generator.isin(generators).to_numpy()
        y_train = rows_train.label.to_numpy()[is_train]
        y_val = rows_val.label.to_numpy()[is_val]
        y_unseen, y_seen = y_eval[is_unseen], y_eval[~is_unseen]

        std = X_train[is_train][:, columns].std(axis=0) if scale else 1
        fold_train, fold_val = X_train[is_train][:, columns] / std, X_val[is_val][:, columns] / std
        fold_eval = X_eval[:, columns] / std

        fold_learner = copy.deepcopy(learner)
        extra = {}
        if getattr(fold_learner, "needs_own", False):
            extra = {"own": own_columns(rows_train[is_train], columns), "own_val": own_columns(rows_val[is_val], columns)}
        fold_learner.fit(fold_train, y_train, fold_val, y_val, **extra)

        threshold = best_threshold(fold_learner.predict(fold_val), y_val)
        unseen, seen = fold_learner.predict(fold_eval[is_unseen]), fold_learner.predict(fold_eval[~is_unseen])
        expert_aucs = [roc_auc_score(y_unseen, fold_eval[is_unseen, c]) for c in range(len(columns))]
        results.append({
            "held out": fold,
            "generators": ", ".join(generators),
            "n_train": int(is_train.sum()),
            "n_val": int(is_val.sum()),
            "n_eval": int(is_unseen.sum()),
            **fold_metrics(unseen, y_unseen, seen, y_seen, threshold),
            "expert_mean_auc": float(np.mean(expert_aucs)),
            "expert_best_auc": float(np.max(expert_aucs)),
        })
    return to_table(results)
