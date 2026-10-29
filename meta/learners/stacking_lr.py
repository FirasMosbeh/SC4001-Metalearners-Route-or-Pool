from sklearn.linear_model import LogisticRegression

from meta.learners.base import MetaLearner


class StackingLRLearner(MetaLearner):
    """Logistic Regression on expert scores (classic stacking).

    Learns one fixed weight per expert identity (Pair A partner to SortedLRLearner).
    """

    def __init__(self, C=1.0, max_iter=1000, random_state=4001):
        self.C = C
        self.max_iter = max_iter
        self.random_state = random_state
        self.model = None

    def fit(self, X, y, X_val=None, y_val=None):
        self.model = LogisticRegression(
            C=self.C,
            max_iter=self.max_iter,
            random_state=self.random_state,
        )
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.decision_function(X)

    @property
    def coef_(self):
        return self.model.coef_ if self.model is not None else None

    @property
    def intercept_(self):
        return self.model.intercept_ if self.model is not None else None

