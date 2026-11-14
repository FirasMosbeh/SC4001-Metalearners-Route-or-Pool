"""Paired bootstrap over the held-out images: is a difference in unseen AUROC larger than chance?

Seeds only cover the combiners' random start; the images are one sample too. Each bootstrap sample redraws,
per fold, the held-out real and fake images with replacement (so every fold keeps its size and balance) and
recomputes the mean unseen AUROC over folds for both sides of a comparison. Both sides are scored on the same
redrawn images, which is what makes it paired. Learners with seeds use the mean AUROC over their seeds, as in
the results table. The experts themselves were trained once, so their randomness is not covered.

The comparisons are fixed here in advance, so they can't be picked after looking at the results.
"""
import numpy as np
import pandas as pd
import joblib
from scipy import sparse

from data import config
from meta.run import BASES, results_path, scores_path

N_SAMPLES = 1000
COMPARISONS = (
    # (question, base, learner, compared with base, learner)
    [("RQ1", base, learner, base, "best_expert") for base in BASES for learner in ["mean", "mlp"]]
    + [("RQ1", base, "generalist", base, learner) for base in BASES for learner in ["mean", "sorted_lr", "mlp", "deep_sets"]]
    + [("RQ2", base, a, base, b) for base in BASES for a, b in [("mlp", "shuffled_mlp"), ("stacking_lr", "sorted_lr")]]
    + [("RQ3", base, learner, base, "mean") for base in BASES for learner in ["stacking_lr", "mlp", "deep_sets", "gating"]]
    + [("RQ4", a, learner, b, learner) for learner in ["mean", "mlp"] for a, b in [("clip", "resnet50"), ("resnet50", "resnet18")]]
    + [("pseudo-unseen", base, "pu_" + learner, base, learner) for base in BASES for learner in ["mlp", "deep_sets"]]
)


class Fold:
    """The held-out images of one fold and N_SAMPLES redraws of them, as counts per image."""

    def __init__(self, labels, rng):
        self.labels = labels
        counts = np.zeros((N_SAMPLES, len(labels)))
        for label in (0, 1):
            images = np.flatnonzero(labels == label)
            draws = rng.integers(0, len(images), size=(N_SAMPLES, len(images)))
            np.add.at(counts, (np.arange(N_SAMPLES)[:, None], images[draws]), 1)
        self.counts = counts

    def aucs(self, scores):
        """AUROC of scores on the full images and on every redraw (Mann-Whitney with counts, ties count half)."""
        values, group = np.unique(scores, return_inverse=True)
        member = sparse.csr_matrix((np.ones(len(scores)), (np.arange(len(scores)), group)), shape=(len(scores), len(values)))
        weights = np.vstack([np.ones(len(scores)), self.counts])
        fake = (weights * self.labels) @ member
        real = (weights * (1 - self.labels)) @ member
        real_below = np.cumsum(real, axis=1) - real
        return (fake * (real_below + real / 2)).sum(axis=1) / (fake.sum(axis=1) * real.sum(axis=1))


def mean_aucs(scores, folds, base, learner, merge_siblings, scale):
    """Mean over folds (and seeds) of the AUROC on the full images, then on every redraw."""
    per_fold = []
    for fold, sample in folds.items():
        runs = [s for (b, l, _, m, sc, f), (labels, s) in scores.items()
                if (b, l, m, sc, f) == (base, learner, merge_siblings, scale, fold)]
        if not runs:
            raise KeyError(f"no scores for {learner} on {base}, fold {fold}")
        per_fold.append(np.mean([sample.aucs(s) for s in runs], axis=0))
    return np.mean(per_fold, axis=0)


def run_bootstrap(eval_pool="test", merge_siblings=True, scale=False, comparisons=COMPARISONS):
    """Difference in mean unseen AUROC (first minus second) with a 95% interval, saved next to the results."""
    scores = joblib.load(scores_path(eval_pool))
    fold_labels = {f: labels for (b, l, _, m, sc, f), (labels, s) in scores.items() if m == merge_siblings and sc == scale}
    rng = np.random.default_rng(config.SEED)
    folds = {fold: Fold(labels, rng) for fold, labels in sorted(fold_labels.items())}
    rows = []
    for question, base_a, learner_a, base_b, learner_b in comparisons:
        difference = (mean_aucs(scores, folds, base_a, learner_a, merge_siblings, scale)
                      - mean_aucs(scores, folds, base_b, learner_b, merge_siblings, scale))
        low, high = np.percentile(difference[1:], [2.5, 97.5])
        rows.append({
            "question": question,
            "comparison": f"{learner_a} ({base_a}) vs {learner_b} ({base_b})",
            "difference": difference[0],
            "low": low,
            "high": high,
            "first_better_share": float((difference[1:] > 0).mean()),
            "clear": bool(low > 0 or high < 0),
        })
    table = pd.DataFrame(rows)
    table.to_csv(results_path(eval_pool).with_name(f"bootstrap_{eval_pool}.csv"), index=False)
    return table


if __name__ == "__main__":
    run_bootstrap()
