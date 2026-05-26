"""Coral health vision classifier (starter).

Classifies coral images into health categories:
  - healthy: normal coloration, intact structure
  - bleached: white/pale coloration, stress indicators
  - dead: algae-covered, structural breakdown

This starter uses a feature-based approach with scikit-learn.
For production, replace with a CNN (ResNet/EfficientNet) fine-tuned
on the CoralNet dataset or AIMS underwater imagery.

Image features extracted:
  - Color histograms (RGB channels)
  - Texture metrics (edge density, variance)
  - Brightness and saturation statistics
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from infrastructure.logging import get_logger

logger = get_logger("models.coral_vision")

CLASSES = ["healthy", "bleached", "dead"]


@dataclass
class VisionResult:
    predicted_class: str
    confidence: float
    class_probabilities: dict[str, float]


def extract_image_features(image: np.ndarray) -> np.ndarray:
    """Extract features from an image array (H, W, 3).

    Returns a 1D feature vector of color + texture statistics.
    """
    if image.ndim == 2:
        image = np.stack([image] * 3, axis=-1)

    features = []
    # Per-channel statistics
    for c in range(3):
        channel = image[:, :, c].astype(float)
        features.extend([
            np.mean(channel),
            np.std(channel),
            np.median(channel),
            np.percentile(channel, 10),
            np.percentile(channel, 90),
        ])

    # Brightness (mean of all channels)
    brightness = np.mean(image, axis=2)
    features.append(float(np.mean(brightness)))
    features.append(float(np.std(brightness)))

    # Saturation proxy (max-min across channels per pixel)
    sat = np.max(image, axis=2).astype(float) - np.min(image, axis=2).astype(float)
    features.append(float(np.mean(sat)))

    # Edge density (simple gradient magnitude)
    gray = np.mean(image, axis=2)
    dy = np.diff(gray, axis=0)
    dx = np.diff(gray, axis=1)
    edge_mag = np.sqrt(dy[:, :-1] ** 2 + dx[:-1, :] ** 2)
    features.append(float(np.mean(edge_mag)))
    features.append(float(np.std(edge_mag)))

    return np.array(features)


def generate_synthetic_training_data(
    n_per_class: int = 100,
    img_size: int = 32,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic coral images for training the starter model.

    Healthy: warm colors (high R, moderate G, low B), moderate texture
    Bleached: high brightness, low saturation, smooth texture
    Dead: dark, green-brown tones, high texture (algae)
    """
    rng = np.random.default_rng(seed)
    X, y = [], []

    for label, class_name in enumerate(CLASSES):
        for _ in range(n_per_class):
            img = np.zeros((img_size, img_size, 3), dtype=np.uint8)
            if class_name == "healthy":
                img[:, :, 0] = rng.integers(150, 220, (img_size, img_size))  # warm R
                img[:, :, 1] = rng.integers(80, 140, (img_size, img_size))   # moderate G
                img[:, :, 2] = rng.integers(40, 100, (img_size, img_size))   # low B
            elif class_name == "bleached":
                base = rng.integers(200, 250, (img_size, img_size))
                img[:, :, 0] = base
                img[:, :, 1] = base - rng.integers(0, 15, (img_size, img_size))
                img[:, :, 2] = base - rng.integers(0, 20, (img_size, img_size))
            else:  # dead
                img[:, :, 0] = rng.integers(40, 90, (img_size, img_size))
                img[:, :, 1] = rng.integers(60, 110, (img_size, img_size))  # greenish
                img[:, :, 2] = rng.integers(30, 70, (img_size, img_size))
                # Add texture noise (algae)
                noise = rng.integers(0, 40, (img_size, img_size, 3), dtype=np.uint8)
                img = np.clip(img.astype(int) + noise, 0, 255).astype(np.uint8)

            X.append(extract_image_features(img))
            y.append(label)

    return np.array(X), np.array(y)


def train_vision_model(
    model_out: str = "models/coral_vision/coral_classifier.joblib",
    n_per_class: int = 100,
) -> dict[str, Any]:
    """Train the coral vision classifier on synthetic data."""
    X, y = generate_synthetic_training_data(n_per_class=n_per_class)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", GradientBoostingClassifier(n_estimators=50, max_depth=4, random_state=42)),
    ])
    model.fit(X_train, y_train)

    accuracy = float(model.score(X_test, y_test))
    out = Path(model_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "classes": CLASSES, "n_features": X.shape[1]}, out)

    logger.info("Coral vision model trained: accuracy=%.3f, saved to %s", accuracy, out)
    return {"accuracy": accuracy, "n_samples": len(X), "n_features": X.shape[1]}


def classify_image(
    model_path: str | Path,
    image: np.ndarray,
) -> VisionResult:
    """Classify a coral image."""
    bundle = joblib.load(model_path)
    model = bundle["model"]
    classes = bundle["classes"]

    features = extract_image_features(image).reshape(1, -1)
    proba = model.predict_proba(features)[0]
    pred_idx = int(np.argmax(proba))

    return VisionResult(
        predicted_class=classes[pred_idx],
        confidence=round(float(proba[pred_idx]), 4),
        class_probabilities={c: round(float(p), 4) for c, p in zip(classes, proba)},
    )
