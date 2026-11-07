import numpy as np

from meta.learners.base import MetaLearner


def hide(X, columns):
    """Copy of X with one expert per image set to 0, the logit of an undecided expert."""
    X = X.copy()
    X[np.arange(len(X)), columns] = 0
    return X


class PseudoUnseenLearner(MetaLearner):
    """Trains any combiner as if each image's generator were unseen (pseudo-unseen meta-training).

    Every training image has its own generator's expert hidden, real images too, so hiding says nothing
    about the label. The combiner can't lean on the expert that knows the image and has to use the others.
    At prediction no expert is the image's own, so each is hidden in turn and the scores are averaged,
    which keeps the input the same as in training (exactly one hidden expert).
    """

    needs_own = True  # fit also takes each image's own expert column, see meta.loo.run_loo

    def __init__(self, base):
        self.base = base

    def fit(self, X, y, X_val=None, y_val=None, own=None, own_val=None):
        self.base.fit(hide(X, own), y, hide(X_val, own_val), y_val)
        return self

    def predict(self, X):
        return np.mean([self.base.predict(hide(X, np.full(len(X), column))) for column in range(X.shape[1])], axis=0)
