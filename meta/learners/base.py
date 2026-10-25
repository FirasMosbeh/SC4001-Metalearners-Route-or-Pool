class MetaLearner:
    """Combines expert scores X (n, experts) into one score per image (> 0 = fake)."""

    def fit(self, X, y, X_val=None, y_val=None):
        """X_val, y_val: validation images (meta_val of the same fold), e.g. for early stopping. May be ignored."""
        return self

    def predict(self, X):
        raise NotImplementedError
