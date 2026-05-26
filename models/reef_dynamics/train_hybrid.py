"""CLI entry point for training the PIML hybrid model."""

from __future__ import annotations

import argparse

from infrastructure.settings import settings
from models.reef_dynamics.hybrid_predictor import train_hybrid_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Physics-Informed ML hybrid model")
    parser.add_argument("--features", default=str(settings.features_path))
    parser.add_argument("--model-out", default="models/reef_dynamics/hybrid_model.joblib")
    args = parser.parse_args()
    train_hybrid_model(args.features, args.model_out)


if __name__ == "__main__":
    main()
