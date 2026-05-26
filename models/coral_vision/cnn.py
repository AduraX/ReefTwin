"""CNN-based coral health classifier using PyTorch.

Uses a lightweight CNN architecture (not full ResNet to keep CPU-friendly).
For production, replace with torchvision ResNet18/EfficientNet fine-tuned
on the CoralNet dataset or AIMS underwater imagery.

Classes: healthy, bleached, dead
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from infrastructure.logging import get_logger

logger = get_logger("models.coral_vision.cnn")

CLASSES = ["healthy", "bleached", "dead"]


@dataclass
class CNNConfig:
    img_size: int = 32
    n_channels: int = 3
    lr: float = 1e-3
    epochs: int = 30
    batch_size: int = 32


class CoralCNN:
    """Lightweight CNN for coral health classification.

    Requires: pip install torch
    """

    def __init__(self, config: CNNConfig | None = None) -> None:
        import torch
        import torch.nn as nn

        self.config = config or CNNConfig()
        self.device = torch.device("cpu")

        self._net = nn.Sequential(
            # Conv block 1: 3 → 16 channels
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # Conv block 2: 16 → 32 channels
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # Conv block 3: 32 → 64 channels
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            # Classifier
            nn.Flatten(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, len(CLASSES)),
        ).to(self.device)

        self._optimizer = torch.optim.Adam(self._net.parameters(), lr=self.config.lr)
        self._loss_fn = nn.CrossEntropyLoss()
        self._trained = False

    def _generate_synthetic_data(self, n_per_class: int = 100):
        """Generate synthetic coral images for training."""
        rng = np.random.default_rng(42)
        images, labels = [], []
        sz = self.config.img_size

        for label, cls in enumerate(CLASSES):
            for _ in range(n_per_class):
                img = np.zeros((sz, sz, 3), dtype=np.float32)
                if cls == "healthy":
                    img[:, :, 0] = rng.uniform(0.55, 0.85, (sz, sz))
                    img[:, :, 1] = rng.uniform(0.30, 0.55, (sz, sz))
                    img[:, :, 2] = rng.uniform(0.15, 0.40, (sz, sz))
                elif cls == "bleached":
                    base = rng.uniform(0.78, 0.98, (sz, sz))
                    img[:, :, 0] = base
                    img[:, :, 1] = base - rng.uniform(0, 0.05, (sz, sz))
                    img[:, :, 2] = base - rng.uniform(0, 0.08, (sz, sz))
                else:  # dead
                    img[:, :, 0] = rng.uniform(0.15, 0.35, (sz, sz))
                    img[:, :, 1] = rng.uniform(0.23, 0.43, (sz, sz))
                    img[:, :, 2] = rng.uniform(0.12, 0.28, (sz, sz))
                    noise = rng.uniform(0, 0.15, (sz, sz, 3)).astype(np.float32)
                    img = np.clip(img + noise, 0, 1)
                images.append(img)
                labels.append(label)

        return np.array(images), np.array(labels)

    def train(self, n_per_class: int = 100) -> dict[str, Any]:
        """Train on synthetic coral images."""
        import torch
        from torch.utils.data import DataLoader, TensorDataset

        images, labels = self._generate_synthetic_data(n_per_class)

        # Shuffle and split
        idx = np.random.permutation(len(images))
        split = int(0.8 * len(idx))
        train_idx, val_idx = idx[:split], idx[split:]

        # Convert to torch: (N, C, H, W)
        def to_tensors(idxs):
            X = torch.tensor(images[idxs].transpose(0, 3, 1, 2), dtype=torch.float32)
            y = torch.tensor(labels[idxs], dtype=torch.long)
            return X, y

        X_train, y_train = to_tensors(train_idx)
        X_val, y_val = to_tensors(val_idx)

        train_loader = DataLoader(
            TensorDataset(X_train, y_train),
            batch_size=self.config.batch_size, shuffle=True,
        )

        for epoch in range(self.config.epochs):
            self._net.train()
            epoch_loss = 0.0
            for xb, yb in train_loader:
                self._optimizer.zero_grad()
                out = self._net(xb)
                loss = self._loss_fn(out, yb)
                loss.backward()
                self._optimizer.step()
                epoch_loss += float(loss) * len(xb)

        # Validation accuracy
        self._net.eval()
        with torch.no_grad():
            val_out = self._net(X_val)
            val_preds = val_out.argmax(dim=1)
            accuracy = float((val_preds == y_val).float().mean())

        self._trained = True
        metrics = {"accuracy": round(accuracy, 4), "epochs": self.config.epochs, "n_samples": len(images)}
        logger.info("Coral CNN trained: accuracy=%.3f (%d samples)", accuracy, len(images))
        return metrics

    def predict(self, image: np.ndarray) -> dict[str, Any]:
        """Classify a coral image (H, W, 3) float32 [0, 1]."""
        import torch

        if image.dtype == np.uint8:
            image = image.astype(np.float32) / 255.0
        if image.ndim == 2:
            image = np.stack([image] * 3, axis=-1)

        # Resize to expected size
        from PIL import Image as _PILImage
        try:
            import PIL
            pil = _PILImage.fromarray((image * 255).astype(np.uint8))
            pil = pil.resize((self.config.img_size, self.config.img_size))
            image = np.array(pil).astype(np.float32) / 255.0
        except ImportError:
            # No PIL — just center-crop/pad
            image = image[:self.config.img_size, :self.config.img_size]

        x = torch.tensor(image.transpose(2, 0, 1), dtype=torch.float32).unsqueeze(0)

        self._net.eval()
        with torch.no_grad():
            out = self._net(x)
            proba = torch.softmax(out, dim=1)[0]

        pred_idx = int(proba.argmax())
        return {
            "predicted_class": CLASSES[pred_idx],
            "confidence": round(float(proba[pred_idx]), 4),
            "class_probabilities": {c: round(float(p), 4) for c, p in zip(CLASSES, proba)},
            "model_type": "cnn",
        }

    def save(self, path: str | Path) -> None:
        import torch
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": self._net.state_dict(), "config": self.config}, path)
        logger.info("Coral CNN saved to %s", path)

    def load(self, path: str | Path) -> None:
        import torch
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self._net.load_state_dict(checkpoint["state_dict"])
        self._trained = True
        logger.info("Coral CNN loaded from %s", path)
