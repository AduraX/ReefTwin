"""AI fairness and explainability metrics.

Addresses NIST AI RMF risk R-009 (model bias).
Provides:
    - Feature importance via permutation importance (no SHAP dependency)
    - Per-reef performance parity check
    - Bias detection across reef subgroups

Uses scikit-learn's built-in tools — no additional dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from infrastructure.logging import get_logger

logger = get_logger("mlops.fairness")


@dataclass
class FairnessReport:
    overall_accuracy: float
    group_metrics: dict[str, dict[str, float]]
    feature_importance: dict[str, float]
    parity_gap: float
    bias_detected: bool
    findings: list[str] = field(default_factory=list)


def compute_feature_importance(
    model_path: str,
    features_path: str,
    n_repeats: int = 5,
) -> dict[str, float]:
    """Compute permutation feature importance."""
    import joblib
    from sklearn.inspection import permutation_importance

    bundle = joblib.load(model_path)
    model = bundle["model"]
    feature_names = bundle["features"]

    from infrastructure.settings import read_df
    df = read_df(features_path)
    X = df[feature_names].ffill().fillna(df[feature_names].median())
    y = df["bleaching_label"].astype(int)

    result = permutation_importance(model, X, y, n_repeats=n_repeats, random_state=42)
    importance = {name: round(float(imp), 4) for name, imp in zip(feature_names, result.importances_mean)}
    return dict(sorted(importance.items(), key=lambda x: -x[1]))


def compute_group_parity(
    model_path: str,
    features_path: str,
    group_column: str = "reef_id",
) -> FairnessReport:
    """Assess prediction fairness across subgroups (reef sites).

    Checks whether the model performs equally well across all reef
    locations, or if it is biased toward/against specific sites.
    """
    import joblib
    from sklearn.metrics import accuracy_score, f1_score

    bundle = joblib.load(model_path)
    model = bundle["model"]
    feature_names = bundle["features"]

    from infrastructure.settings import read_df
    df = read_df(features_path)
    X = df[feature_names].ffill().fillna(df[feature_names].median())
    y = df["bleaching_label"].astype(int)
    groups = df[group_column]

    y_pred = model.predict(X)
    overall_acc = float(accuracy_score(y, y_pred))

    group_metrics = {}
    accuracies = []

    for group_name in groups.unique():
        mask = groups == group_name
        if mask.sum() < 2:
            continue
        g_y = y[mask]
        g_pred = y_pred[mask]
        acc = float(accuracy_score(g_y, g_pred))
        f1 = float(f1_score(g_y, g_pred, zero_division=0))
        n_samples = int(mask.sum())
        pos_rate = float(g_y.mean())
        pred_pos_rate = float(g_pred.mean())

        group_metrics[group_name] = {
            "accuracy": round(acc, 4),
            "f1_score": round(f1, 4),
            "n_samples": n_samples,
            "actual_positive_rate": round(pos_rate, 4),
            "predicted_positive_rate": round(pred_pos_rate, 4),
        }
        accuracies.append(acc)

    # Parity gap = max accuracy difference across groups
    parity_gap = max(accuracies) - min(accuracies) if len(accuracies) >= 2 else 0.0

    # Feature importance
    importance = compute_feature_importance(model_path, features_path)

    # Findings
    findings = []
    if parity_gap > 0.15:
        worst = min(group_metrics.items(), key=lambda x: x[1]["accuracy"])
        best = max(group_metrics.items(), key=lambda x: x[1]["accuracy"])
        findings.append(
            f"Significant accuracy gap ({parity_gap:.1%}) between {best[0]} ({best[1]['accuracy']:.1%}) "
            f"and {worst[0]} ({worst[1]['accuracy']:.1%})"
        )
    if parity_gap > 0.05:
        findings.append("Model performance varies across reef sites — review training data distribution")

    for g, m in group_metrics.items():
        if abs(m["actual_positive_rate"] - m["predicted_positive_rate"]) > 0.2:
            findings.append(f"Prediction bias for {g}: actual positive rate {m['actual_positive_rate']:.1%} vs predicted {m['predicted_positive_rate']:.1%}")

    if not findings:
        findings.append("No significant fairness concerns detected across reef subgroups")

    bias_detected = parity_gap > 0.15 or any("bias" in f.lower() for f in findings)

    report = FairnessReport(
        overall_accuracy=round(overall_acc, 4),
        group_metrics=group_metrics,
        feature_importance=importance,
        parity_gap=round(parity_gap, 4),
        bias_detected=bias_detected,
        findings=findings,
    )

    logger.info(
        "Fairness report: accuracy=%.3f parity_gap=%.3f bias=%s",
        overall_acc, parity_gap, bias_detected,
    )
    return report
