import copy
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from meta.learners.base import MetaLearner


class MLPNet(nn.Module):
    def __init__(self, in_features, hidden_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


class MLPLearner(MetaLearner):
    """Small neural network trained on expert scores (routing combiner).

    Learns non-linear interactions across specific experts (Pair B partner to ShuffledMLP).
    """

    def __init__(self, hidden_dim=32, lr=1e-2, weight_decay=1e-4, epochs=20, batch_size=256, seed=4001):
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.batch_size = batch_size
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

