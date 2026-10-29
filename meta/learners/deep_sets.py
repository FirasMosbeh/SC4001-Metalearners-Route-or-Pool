import copy
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from meta.learners.base import MetaLearner


class DeepSetsNet(nn.Module):
    """Deep Sets network: phi(score) element-wise -> sum/pool -> rho(set_embed) -> logit."""

    def __init__(self, hidden_dim=32, pool="sum"):
        super().__init__()
        self.pool = pool
        self.phi = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.rho = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        # x: (batch_size, n_experts) -> (batch_size, n_experts, 1)
        x_items = x.unsqueeze(-1)
        # phi applied to each score individually: (batch_size, n_experts, hidden_dim)
        embeddings = self.phi(x_items)

        # Permutation-invariant set aggregation
        if self.pool == "mean":
            set_repr = embeddings.mean(dim=1)
        else:
            set_repr = embeddings.sum(dim=1)

        # Decide from set representation
        return self.rho(set_repr).squeeze(-1)


class DeepSetsLearner(MetaLearner):
    """Deep Sets combiner (pooling): permutation-invariant neural net over expert scores.

    Applies the same small network to each score individually, sums across experts,
    and decides. Inherently blind to expert identity and handles missing experts.
    """

    def __init__(self, hidden_dim=32, pool="sum", lr=1e-2, weight_decay=1e-4, epochs=20, batch_size=256, seed=4001):
        self.hidden_dim = hidden_dim
        self.pool = pool
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.batch_size = batch_size
        self.seed = seed
        self.model = None

    def fit(self, X, y, X_val=None, y_val=None):
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        self.model = DeepSetsNet(hidden_dim=self.hidden_dim, pool=self.pool)
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
                optimizer.zero_grad()
                out = self.model(bx)
                loss = criterion(out, by)
                loss.backward()
                optimizer.step()

            if has_val:
                self.model.eval()
                with torch.no_grad():
                    val_out = self.model(X_val_t)
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
            scores = self.model(X_tensor)
            return scores.cpu().numpy()

