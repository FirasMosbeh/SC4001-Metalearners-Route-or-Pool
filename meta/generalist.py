"""One detector instead of experts: a single CLIP probe per fold, the baseline for RQ1.

Same recipe as the CLIP experts (clip.train) and the same training images: the expert_train images of
every seen expert together, so the only difference to the experts is one model instead of several.
Evaluated like a meta-learner in meta.loo, with the threshold from the seen meta_val images.
"""
from functools import lru_cache

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from data.features import load_features
from data.images import load_manifest
from meta.loo import best_threshold, fold_metrics, folds, to_table
from resnet.experts import EXPERTS, expert_rows

FEATURE_NAME = "clip_vit-l14"  # clip.extract.FEATURE_NAME, which would import transformers
MAX_ITER = 2000  # as in clip.train


@lru_cache(maxsize=None)
def load_pool(pool):
    return load_features(FEATURE_NAME, pool=pool)


@lru_cache(maxsize=None)
def fit_generalist(held_out):
    """Probe on the expert_train images of every expert not in held_out (a tuple of experts)."""
    manifest = load_manifest()
    rows = pd.concat([expert_rows(manifest, expert, "expert_train") for expert in EXPERTS if expert not in held_out])
    rows = rows.drop_duplicates("file")
    X, pool_rows = load_pool("expert_train")
    positions = pd.Series(range(len(pool_rows)), index=pool_rows.file.to_numpy())
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=MAX_ITER))
    return model.fit(X[positions[rows.file].to_numpy()], rows.label.to_numpy()), len(rows)


def run_generalist(merge_siblings=False, eval_pool="test", scores=None):
    """One row per fold plus the mean, with the same columns as meta.loo.run_loo (scores as there too)."""
    X_val, rows_val = load_pool("meta_val")
    X_eval, rows_eval = load_pool(eval_pool)
    y_eval = rows_eval.label.to_numpy()
    results = []
    for fold, experts in folds(merge_siblings):
        generators = [generator for expert in experts for generator in EXPERTS[expert]]
        is_val = ~rows_val.generator.isin(generators).to_numpy()
        is_unseen = rows_eval.generator.isin(generators).to_numpy()
        model, n_train = fit_generalist(tuple(experts))
        y_val = rows_val.label.to_numpy()[is_val]
        threshold = best_threshold(model.decision_function(X_val[is_val]), y_val)
        unseen = model.decision_function(X_eval[is_unseen])
        if scores is not None:
            scores[fold] = (y_eval[is_unseen].astype("int8"), unseen.astype("float32"))
        results.append({
            "held out": fold,
            "generators": ", ".join(generators),
            "n_train": n_train,
            "n_val": int(is_val.sum()),
            "n_eval": int(is_unseen.sum()),
            **fold_metrics(unseen, y_eval[is_unseen],
                           model.decision_function(X_eval[~is_unseen]), y_eval[~is_unseen], threshold),
        })
    return to_table(results)
