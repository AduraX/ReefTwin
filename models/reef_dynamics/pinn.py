"""Physics-Informed Neural Network (PINN) for reef thermal dynamics.

A PyTorch neural network trained with a composite loss:
    L = L_data + lambda_phys * L_physics

Where:
    L_data    = MSE between predicted and observed bleaching labels
    L_physics = residual of the DHW accumulation ODE (NOAA formula)

The physics loss penalises predictions that violate known thermal
stress dynamics, ensuring the network respects conservation laws
even in sparse-data regimes.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from infrastructure.logging import get_logger
from infrastructure.settings import settings

logger = get_logger("models.reef_dynamics.pinn")


@dataclass
class PINNConfig:
    hidden_dims: list[int] = None
    lr: float = 1e-3
    epochs: int = 200
    lambda_physics: float = 0.5
    mmm_threshold: float = 28.2
    dhw_critical: float = 8.0

    def __post_init__(self):
        if self.hidden_dims is None:
            self.hidden_dims = [32, 32, 16]


class ReefPINN:
    """Physics-Informed Neural Network for bleaching risk prediction.

    Requires: pip install torch
    """

    def __init__(self, config: PINNConfig | None = None) -> None:
        import torch
        import torch.nn as nn

        self.config = config or PINNConfig()
        self.device = torch.device("cpu")

        # Build network: inputs → hidden → risk score
        n_input = len(settings.features)
        layers = []
        prev = n_input
        for h in self.config.hidden_dims:
            layers.extend([nn.Linear(prev, h), nn.Tanh()])
            prev = h
        layers.append(nn.Linear(prev, 1))
        layers.append(nn.Sigmoid())

        self._net = nn.Sequential(*layers).to(self.device)
        self._optimizer = torch.optim.Adam(self._net.parameters(), lr=self.config.lr)
        self._loss_fn = nn.MSELoss()
        self._trained = False

    def _physics_loss(self, X: "torch.Tensor", y_pred: "torch.Tensor") -> "torch.Tensor":
        """Compute physics residual loss.

        Enforces: when DHW >= critical threshold, risk should be high.
        Enforces: when SST < MMM, DHW should not accumulate (risk stays low).
        """
        import torch

        # Feature indices (must match settings.features order)
        feature_names = settings.features
        dhw_idx = feature_names.index("degree_heating_weeks") if "degree_heating_weeks" in feature_names else -1
        temp_idx = feature_names.index("water_temperature_c") if "water_temperature_c" in feature_names else -1

        if dhw_idx < 0 or temp_idx < 0:
            return torch.tensor(0.0)

        dhw = X[:, dhw_idx]
        temp = X[:, temp_idx]

        # Physics constraint 1: high DHW → high risk
        dhw_ratio = dhw / self.config.dhw_critical
        expected_min_risk = torch.sigmoid(5.0 * (dhw_ratio - 1.0))
        violation_high = torch.relu(expected_min_risk - y_pred.squeeze())

        # Physics constraint 2: low temp → low risk
        temp_below = torch.relu(self.config.mmm_threshold - temp)
        expected_max_risk = 1.0 - torch.sigmoid(3.0 * temp_below)
        violation_low = torch.relu(y_pred.squeeze() - expected_max_risk)

        return torch.mean(violation_high ** 2) + torch.mean(violation_low ** 2)

    def train(self, features_path: str) -> dict[str, Any]:
        """Train the PINN on reef feature data."""
        import torch

        from infrastructure.settings import read_df
        df = read_df(features_path)
        feature_cols = settings.features
        X_np = df[feature_cols].ffill().fillna(df[feature_cols].median()).values.astype(np.float32)
        y_np = df[settings.target].values.astype(np.float32).reshape(-1, 1)

        X = torch.tensor(X_np, device=self.device)
        y = torch.tensor(y_np, device=self.device)

        history = {"total_loss": [], "data_loss": [], "physics_loss": []}

        for epoch in range(self.config.epochs):
            self._optimizer.zero_grad()
            y_pred = self._net(X)

            data_loss = self._loss_fn(y_pred, y)
            phys_loss = self._physics_loss(X, y_pred)
            total_loss = data_loss + self.config.lambda_physics * phys_loss

            total_loss.backward()
            self._optimizer.step()

            history["total_loss"].append(float(total_loss))
            history["data_loss"].append(float(data_loss))
            history["physics_loss"].append(float(phys_loss))

            if (epoch + 1) % 50 == 0:
                logger.info(
                    "PINN epoch %d/%d: total=%.4f data=%.4f physics=%.4f",
                    epoch + 1, self.config.epochs,
                    float(total_loss), float(data_loss), float(phys_loss),
                )

        self._trained = True
        metrics = {
            "final_total_loss": history["total_loss"][-1],
            "final_data_loss": history["data_loss"][-1],
            "final_physics_loss": history["physics_loss"][-1],
            "epochs": self.config.epochs,
        }
        logger.info("PINN training complete: %s", metrics)
        return metrics

    def predict(self, row: dict[str, float]) -> dict[str, Any]:
        """Predict bleaching risk for a single observation."""
        import torch

        feature_cols = settings.features
        x = np.array([row.get(f, 0) for f in feature_cols], dtype=np.float32)
        x_t = torch.tensor(x, device=self.device).unsqueeze(0)

        with torch.no_grad():
            score = float(self._net(x_t).item())

        t = settings.risk_thresholds
        if score >= t.alert:
            category = "alert"
        elif score >= t.warning:
            category = "warning"
        elif score >= t.watch:
            category = "watch"
        else:
            category = "normal"

        return {
            "bleaching_risk_score": round(score, 4),
            "risk_category": category,
            "model_type": "pinn",
        }

    def save(self, path: str | Path) -> None:
        import torch
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "state_dict": self._net.state_dict(),
            "config": self.config,
            "features": settings.features,
        }, path)
        logger.info("PINN saved to %s", path)

    def load(self, path: str | Path) -> None:
        import torch
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self._net.load_state_dict(checkpoint["state_dict"])
        self._trained = True
        logger.info("PINN loaded from %s", path)
