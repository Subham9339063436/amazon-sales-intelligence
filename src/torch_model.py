"""PyTorch neural-network forecaster for the monthly product panel.

Design constraints (Phase 3):
* Consumes the exact leakage-free feature matrix and target produced by
  ``src.data.build_monthly_panel`` + ``src.features.add_lag_features``
  (target definition unchanged from Phase 1).
* Evaluated with the same chronological holdout as Phase 1
  (``src.split.chronological_split``); early stopping only ever sees a tail
  carved from the *training* period — never the test set.
* All standardization statistics are computed on the training partition only.
* Seeded end-to-end (python / numpy / torch) for reproducible runs on CPU.
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .config import (
    RANDOM_SEED,
    TORCH_HIDDEN,
    TORCH_LEARNING_RATE,
    TORCH_MAX_EPOCHS,
    TORCH_PATIENCE,
    TORCH_WEIGHT_DECAY,
)


def set_seed(seed: int) -> None:
    """Seed python, numpy and torch for reproducible training runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class _MLP(nn.Module):
    """Feed-forward regressor: hidden ReLU blocks -> single linear output."""

    def __init__(self, n_features: int, hidden: tuple[int, ...] = TORCH_HIDDEN):
        super().__init__()
        dims = [n_features, *hidden]
        layers: list[nn.Module] = []
        for in_dim, out_dim in zip(dims[:-1], dims[1:]):
            layers.extend([nn.Linear(in_dim, out_dim), nn.ReLU()])
        layers.append(nn.Linear(dims[-1], 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class MLPForecaster:
    """Sklearn-like wrapper: ``fit`` / ``predict`` / ``save`` / ``load``.

    Features and target are standardized with statistics fit on the training
    partition only; predictions are inverse-transformed back to dollars.
    """

    def __init__(
        self,
        seed: int = RANDOM_SEED,
        hidden: tuple[int, ...] = TORCH_HIDDEN,
        learning_rate: float = TORCH_LEARNING_RATE,
        weight_decay: float = TORCH_WEIGHT_DECAY,
        max_epochs: int = TORCH_MAX_EPOCHS,
        patience: int = TORCH_PATIENCE,
    ) -> None:
        self.seed = seed
        self.hidden = tuple(hidden)
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.max_epochs = max_epochs
        self.patience = patience
        self.model: _MLP | None = None
        self.x_mean: np.ndarray | None = None
        self.x_std: np.ndarray | None = None
        self.y_mean: float | None = None
        self.y_std: float | None = None
        self.training_info: dict = {}

    # ------------------------------------------------------------------ utils
    def _standardize_x(self, X: np.ndarray) -> np.ndarray:
        assert self.x_mean is not None and self.x_std is not None
        return ((X - self.x_mean) / self.x_std).astype(np.float32)

    def _standardize_y(self, y: np.ndarray) -> np.ndarray:
        assert self.y_mean is not None and self.y_std is not None
        return ((y - self.y_mean) / self.y_std).astype(np.float32)

    def _denormalize_y(self, y_std: np.ndarray) -> np.ndarray:
        assert self.y_mean is not None and self.y_std is not None
        return (y_std * self.y_std + self.y_mean).astype(np.float64)

    @property
    def is_fitted(self) -> bool:
        return self.model is not None and self.x_mean is not None

    # -------------------------------------------------------------------- API
    def fit(
        self,
        X_train,
        y_train,
        X_val=None,
        y_val=None,
    ) -> "MLPForecaster":
        """Train with Adam/MSE.

        With a validation split (a tail of the *training* period), training
        early-stops on validation MSE and restores the best weights. Without
        one, a fixed number of epochs is run. Test data must never be passed
        here as validation.
        """
        X_train = np.asarray(X_train, dtype=np.float64)
        y_train = np.asarray(y_train, dtype=np.float64)
        if X_train.ndim != 2:
            raise ValueError("X_train must be a 2-D array.")
        if len(X_train) != len(y_train):
            raise ValueError("X_train and y_train must have the same length.")
        if X_val is not None:
            X_val = np.asarray(X_val, dtype=np.float64)
            y_val = np.asarray(y_val, dtype=np.float64)
            if len(X_val) != len(y_val):
                raise ValueError("X_val and y_val must have the same length.")

        set_seed(self.seed)

        # Train-only standardization statistics (no leakage).
        self.x_mean = X_train.mean(axis=0)
        self.x_std = X_train.std(axis=0)
        self.x_std[self.x_std == 0.0] = 1.0
        self.y_mean = float(y_train.mean())
        self.y_std = float(y_train.std())
        if not np.isfinite(self.y_std) or self.y_std < 1e-12:
            self.y_std = 1.0

        xs = torch.tensor(self._standardize_x(X_train))
        ys = torch.tensor(self._standardize_y(y_train))
        xv = yv = None
        if X_val is not None:
            xv = torch.tensor(self._standardize_x(X_val))
            yv = torch.tensor(self._standardize_y(y_val))

        self.model = _MLP(X_train.shape[1], self.hidden)
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        mse = nn.MSELoss()

        best_state = None
        best_loss = float("inf")
        best_epoch = 0
        wait = 0
        epochs_run = 0

        for epoch in range(1, self.max_epochs + 1):
            self.model.train()
            optimizer.zero_grad()
            loss = mse(self.model(xs), ys)
            loss.backward()
            optimizer.step()
            epochs_run = epoch

            if xv is not None:
                self.model.eval()
                with torch.no_grad():
                    monitor = float(mse(self.model(xv), yv))
            else:
                monitor = float(loss.detach())

            if monitor < best_loss - 1e-8:
                best_loss = monitor
                best_epoch = epoch
                wait = 0
                best_state = {
                    k: v.detach().clone() for k, v in self.model.state_dict().items()
                }
            else:
                wait += 1
                if xv is not None and wait >= self.patience:
                    break

        if best_state is not None:
            self.model.load_state_dict(best_state)
        self.model.eval()

        self.training_info = {
            "epochs_run": epochs_run,
            "best_epoch": best_epoch,
            "best_monitor": best_loss,  # MSE in normalized target space
            "early_stopping": xv is not None,
            "n_train_rows": int(len(X_train)),
            "n_val_rows": int(len(X_val)) if X_val is not None else 0,
            "seed": self.seed,
            "hidden": list(self.hidden),
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "max_epochs": self.max_epochs,
            "patience": self.patience,
            "device": "cpu",
        }
        return self

    def predict(self, X) -> np.ndarray:
        """Predict in original target units; return shape is always ``(n,)``."""
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted yet.")
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if X.ndim != 2:
            raise ValueError("X must be 1-D or 2-D.")
        if X.shape[1] != self.x_mean.shape[0]:
            raise ValueError(
                f"Expected {self.x_mean.shape[0]} features, got {X.shape[1]}."
            )
        assert self.model is not None
        self.model.eval()
        with torch.no_grad():
            z = self.model(torch.tensor(self._standardize_x(X))).numpy()
        return self._denormalize_y(z)

    def save(self, path) -> None:
        """Persist weights, scalers, hyperparameters and training metadata."""
        if not self.is_fitted:
            raise RuntimeError("Cannot save an unfitted model.")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": self.model.state_dict(),
                "n_features": int(self.x_mean.shape[0]),
                "hidden": self.hidden,
                "seed": self.seed,
                "learning_rate": self.learning_rate,
                "weight_decay": self.weight_decay,
                "max_epochs": self.max_epochs,
                "patience": self.patience,
                "x_mean": self.x_mean,
                "x_std": self.x_std,
                "y_mean": self.y_mean,
                "y_std": self.y_std,
                "training_info": self.training_info,
            },
            path,
        )

    @classmethod
    def load(cls, path) -> "MLPForecaster":
        """Rebuild a fitted forecaster from a checkpoint written by ``save``."""
        ckpt = torch.load(Path(path), map_location="cpu", weights_only=False)
        obj = cls(
            seed=int(ckpt["seed"]),
            hidden=tuple(ckpt["hidden"]),
            learning_rate=float(ckpt["learning_rate"]),
            weight_decay=float(ckpt["weight_decay"]),
            max_epochs=int(ckpt["max_epochs"]),
            patience=int(ckpt["patience"]),
        )
        obj.model = _MLP(int(ckpt["n_features"]), tuple(ckpt["hidden"]))
        obj.model.load_state_dict(ckpt["state_dict"])
        obj.model.eval()
        obj.x_mean = ckpt["x_mean"]
        obj.x_std = ckpt["x_std"]
        obj.y_mean = float(ckpt["y_mean"])
        obj.y_std = float(ckpt["y_std"])
        obj.training_info = dict(ckpt.get("training_info", {}))
        return obj
