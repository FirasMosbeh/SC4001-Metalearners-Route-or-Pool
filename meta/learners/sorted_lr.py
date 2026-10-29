import numpy as np
from sklearn.linear_model import LogisticRegression

from meta.learners.base import MetaLearner


class SortedLRLearner(MetaLearner):
    """Sorts each image's expert scores high -> low, then fits Logistic Regression.

    Learns weights for score ranks (e.g. how much the top score matters)
    while completely ignoring expert identity (pooling control for stacking LR).
    """

    def __init__(self, C=1.0, max_iter=1000, random_state=4001):
        self.C = C
        self.max_iter = max_iter
        self.random_state = random_state
        self.model = None

    def _sort(self, X):
        # Sort descending per image: highest logit to lowest logit
        return np.sort(X, axis=1)[:, ::-1]

    def fit(self, X, y, X_val=None, y_val=None):
        X_sorted = self._sort(X)
        self.model = LogisticRegression(
            C=self.C,
            max_iter=self.max_iter,
            random_state=self.random_state,
        )
        self.model.fit(X_sorted, y)
        return self

    def predict(self, X):
        X_sorted = self._sort(X)
        return self.model.decision_function(X_sorted)

    @property
    def coef_(self):
        return self.model.coef_ if self.model is not None else None

    @property
    def intercept_(self):
        return self.model.intercept_ if self.model is not None else None

