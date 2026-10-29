from meta.learners.base import MetaLearner


class MeanLearner(MetaLearner):
    """Average of expert logits. Nothing to fit."""

    def predict(self, X):
        return X.mean(axis=1)
