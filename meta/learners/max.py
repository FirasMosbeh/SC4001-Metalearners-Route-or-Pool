from meta.learners.base import MetaLearner


class MaxLearner(MetaLearner):
    """Most confident 'fake' vote: the highest expert logit. Nothing to fit."""

    def predict(self, X):
        return X.max(axis=1)
