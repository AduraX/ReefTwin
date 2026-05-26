from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from infrastructure.logging import get_logger
from infrastructure.settings import read_df, settings

logger = get_logger("models.bleaching_risk.train")


def train_model(features_path: str, model_out: str) -> dict:
    df = read_df(features_path)

    feature_cols = settings.features
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required feature columns: {missing}")

    X = df[feature_cols]
    # Forward-fill time-series gaps, then median for remaining NaN (not zeros)
    X = X.ffill().fillna(X.median())

    y = df[settings.target].astype(int)
    stratify = y if y.nunique() > 1 and y.value_counts().min() > 1 else None
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=stratify)
    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(n_estimators=120, random_state=42, class_weight="balanced")),
        ]
    )
    model.fit(X_train, y_train)

    prob_matrix = model.predict_proba(X_test)
    classes = list(model.named_steps["clf"].classes_)
    if 1 in classes:
        proba = prob_matrix[:, classes.index(1)]
    else:
        logger.warning("Positive class (1) not in model classes %s", classes)
        proba = prob_matrix[:, 0] * 0

    preds = (proba >= 0.5).astype(int)
    metrics = {
        "roc_auc": float(roc_auc_score(y_test, proba)) if y_test.nunique() > 1 else None,
        "classification_report": classification_report(y_test, preds, output_dict=True, zero_division=0),
    }
    out = Path(model_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "features": feature_cols, "metrics": metrics}, out)
    logger.info("Model saved to %s | ROC-AUC: %s", out, metrics["roc_auc"])

    # MLflow experiment tracking
    from infrastructure.mlops.tracking import tracked_experiment
    with tracked_experiment(
        "bleaching_risk_training",
        run_name=f"rf_{out.stem}",
        params={"n_estimators": 120, "class_weight": "balanced", "features": str(feature_cols)},
        tags={"model_type": "random_forest", "target": settings.target},
    ) as run:
        if metrics.get("roc_auc") is not None:
            run.log_metric("roc_auc", metrics["roc_auc"])
        report = metrics.get("classification_report", {})
        if isinstance(report, dict) and "weighted avg" in report:
            run.log_metric("f1_weighted", report["weighted avg"].get("f1-score", 0))
            run.log_metric("precision_weighted", report["weighted avg"].get("precision", 0))
            run.log_metric("recall_weighted", report["weighted avg"].get("recall", 0))
        run.log_artifact(str(out))

    # Generate model card
    from infrastructure.mlops.governance import create_bleaching_model_card
    card = create_bleaching_model_card(
        {k: v for k, v in metrics.items() if isinstance(v, (int, float)) and v is not None}
    )
    card_path = out.parent / "model_card.json"
    card.save(card_path)

    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", default=str(settings.features_path))
    parser.add_argument("--model-out", default=str(settings.model_path))
    args = parser.parse_args()
    train_model(args.features, args.model_out)


if __name__ == "__main__":
    main()
