"""Leave-one-out evaluation of a meta-learner on the scores of one expert pool (the base).

One fold per expert: its generators are held out. The learner is fitted on the meta_train images of the
other generators, without the held-out expert's column, and evaluated on the test images of the held-out
generators, which neither the experts nor the learner have seen. meta_val of the other generators is
passed to fit as validation.
"""
import copy

import pandas as pd
from sklearn.metrics import roc_auc_score

from clip.score import SCORES_NAME
from data.features import load_features
from resnet.experts import EXPERTS
from resnet.score import feature_name

# base -> saved expert scores (one raw logit per expert, columns in EXPERTS order)
BASES = {
    "clip": SCORES_NAME,
    "resnet18": feature_name("resnet18"),
    "resnet50": feature_name("resnet50"),
}


def folds():
    """(held-out expert, its generators), one per expert."""
    for expert, generators in EXPERTS.items():
        yield expert, generators


def run_loo(learner, base):
    """Fit and evaluate a fresh copy of learner per fold. Returns one row per fold plus the mean."""
    X_train, rows_train = load_features(BASES[base], pool="meta_train")
    X_val, rows_val = load_features(BASES[base], pool="meta_val")
    X_test, rows_test = load_features(BASES[base], pool="test")
    results = []
    for column, (expert, generators) in enumerate(folds()):
        columns = [c for c in range(len(EXPERTS)) if c != column]
        is_train = ~rows_train.generator.isin(generators).to_numpy()
        is_val = ~rows_val.generator.isin(generators).to_numpy()
        is_test = rows_test.generator.isin(generators).to_numpy()
        y_train = rows_train.label.to_numpy()[is_train]
        y_val = rows_val.label.to_numpy()[is_val]
        y_test = rows_test.label.to_numpy()[is_test]

        fold_learner = copy.deepcopy(learner).fit(X_train[is_train][:, columns], y_train,
                                                  X_val[is_val][:, columns], y_val)
        scores = fold_learner.predict(X_test[is_test][:, columns])
        results.append({
            "held out": expert,
            "generators": ", ".join(generators),
            "n_train": int(is_train.sum()),
            "n_val": int(is_val.sum()),
            "n_test": int(is_test.sum()),
            "auc": roc_auc_score(y_test, scores),
            "acc": float(((scores > 0) == (y_test == 1)).mean()),
        })
    table = pd.DataFrame(results).set_index("held out")
    table.loc["mean", ["auc", "acc"]] = table[["auc", "acc"]].mean()
    return table
