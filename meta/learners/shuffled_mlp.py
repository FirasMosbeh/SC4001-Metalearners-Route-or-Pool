import copy
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from meta.learners.base import MetaLearner
from meta.learners.mlp import MLPNet


class ShuffledMLPLearner(MetaLearner):
    """The MLP from #7, with expert column order randomly shuffled every batch (control).

    Has the same capacity and size as MLP, but is forced to be blind to expert identity
    (Pair B control partner to MLPLearner).
    """

    def __init__(self, hidden_dim=32, lr=1e-2, weight_decay=1e-4, epochs=20, batch_size=256, test_permutations=10, seed=4001):
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.batch_size = batch_size
        self.test_permutations = test_permutations
        self.seed = seed
        self.model = None

    def fit(self, X, y, X_val=None, y_val=None):
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        in_features = X.shape[1]
        self.model = MLPNet(in_features=in_features, hidden_dim=self.hidden_dim)

        device = torch.device("cpu")
        self.model.to(device)

        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        X_tensor = torch.as_tensor(X, dtype=torch.float32)
        y_tensor = torch.as_tensor(y, dtype=torch.float32)
        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        has_val = X_val is not None and y_val is not None
        if has_val:
            X_val_t = torch.as_tensor(X_val, dtype=torch.float32).to(device)
            y_val_t = torch.as_tensor(y_val, dtype=torch.float32).to(device)

        best_val_loss = float("inf")
        best_state = copy.deepcopy(self.model.state_dict())

        for epoch in range(self.epochs):
            self.model.train()
            for bx, by in loader:
                bx, by = bx.to(device), by.to(device)

                # Shuffle expert order randomly for every batch (destroys identity)
                perm = torch.randperm(bx.shape[1], device=device)
                bx = bx[:, perm]

                optimizer.zero_grad()
                out = self.model(bx)
                loss = criterion(out, by)
                loss.backward()
                optimizer.step()

            if has_val:
                self.model.eval()
                with torch.no_grad():
                    # Evaluate on validation with random permutations
                    val_preds = []
                    for _ in range(5):
                        p = torch.randperm(X_val_t.shape[1], device=device)
                        val_preds.append(self.model(X_val_t[:, p]))
                    val_out = torch.stack(val_preds, dim=0).mean(dim=0)
                    val_loss = criterion(val_out, y_val_t).item()
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        best_state = copy.deepcopy(self.model.state_dict())

        if has_val:
            self.model.load_state_dict(best_state)

        return self

    def predict(self, X):
        self.model.eval()
        device = next(self.model.parameters()).device
        with torch.no_grad():
            X_tensor = torch.as_tensor(X, dtype=torch.float32, device=device)
            preds = []
            for _ in range(self.test_permutations):
                perm = torch.randperm(X_tensor.shape[1], device=device)
                preds.append(self.model(X_tensor[:, perm]))
            scores = torch.stack(preds, dim=0).mean(dim=0)
            return scores.cpu().numpy()

