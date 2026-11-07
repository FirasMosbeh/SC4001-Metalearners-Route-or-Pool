from sklearn.metrics import roc_auc_score

from meta.learners.base import MetaLearner


class BestExpertLearner(MetaLearner):
    """The single expert with the highest AUROC on the seen validation images (RQ1 baseline)."""

    def fit(self, X, y, X_val=None, y_val=None):
        self.column = max(range(X_val.shape[1]), key=lambda column: roc_auc_score(y_val, X_val[:, column]))
        return self

    def predict(self, X):
        return X[:, self.column]
