from __future__ import annotations

import argparse
from datetime import datetime, timezone

import pandas as pd

from infrastructure.db.factory import get_state_store
from infrastructure.logging import get_logger
from infrastructure.settings import read_df, settings
from models.bleaching_risk.inference import predict_risk

logger = get_logger("pipelines.update_twin_state")


def ecosystem_status(risk_category: str) -> str:
    return {
        "normal": "stable",
        "watch": "watch",
        "warning": "stressed",
        "alert": "critical",
    }.get(risk_category, "unknown")


def update_twin_state(features_path: str, model_path: str) -> dict:
    df = read_df(features_path)
    if df.empty:
        logger.warning("Features file is empty: %s", features_path)
        return {"generated_at": datetime.now(timezone.utc).isoformat(), "states": []}

    latest = df.sort_values("date").groupby("reef_id", as_index=False).tail(1)
    states = []
    for _, row in latest.iterrows():
        record = row.to_dict()
        try:
            pred = predict_risk(model_path, record)
        except FileNotFoundError:
            logger.error("Model not found at %s — run 'make train-model' first", model_path)
            raise
        state = {
            "reef_id": record["reef_id"],
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "sst_celsius": round(float(record.get("sst_celsius", 0)), 3),
            "water_temperature_c": round(float(record.get("water_temperature_c", 0)), 3),
            "ph": round(float(record.get("ph", 0)), 3),
            "salinity_psu": round(float(record.get("salinity_psu", 0)), 3),
            "turbidity_ntu": round(float(record.get("turbidity_ntu", 0)), 3),
            "dissolved_oxygen_mg_l": round(float(record.get("dissolved_oxygen_mg_l", 0)), 3),
            "degree_heating_weeks": round(float(record.get("degree_heating_weeks", 0)), 3),
            "bleaching_risk_score": pred["bleaching_risk_score"],
            "risk_category": pred["risk_category"],
            "ecosystem_status": ecosystem_status(pred["risk_category"]),
        }
        states.append(state)

    payload = {"generated_at": datetime.now(timezone.utc).isoformat(), "states": states}

    store = get_state_store()
    store.save(payload)
    logger.info("Wrote twin state for %d reefs via %s backend", len(states), settings.state_store_backend)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", default=str(settings.features_path))
    parser.add_argument("--model", default=str(settings.model_path))
    args = parser.parse_args()
    update_twin_state(args.features, args.model)


if __name__ == "__main__":
    main()
