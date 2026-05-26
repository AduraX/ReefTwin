from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from infrastructure.logging import get_logger
from infrastructure.settings import settings

logger = get_logger("pipelines.ingest_noaa_crw")


def generate_noaa_sample(days: int = 60, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    reef_ids = settings.reef_ids
    today = datetime.now(timezone.utc).date()
    records = []
    for reef_id in reef_ids:
        heat_bias = 1.2 if reef_id == reef_ids[0] else 0.3
        for d in range(days):
            date = today - timedelta(days=days - d - 1)
            sst = rng.normal(28.5, 0.5) + (heat_bias if d > days * 0.65 else 0)
            anomaly = sst - 28.2
            hotspot = max(0.0, anomaly - 0.7)
            dhw = max(0.0, hotspot * min(8, d / 7))
            alert = "normal"
            if dhw >= 8:
                alert = "alert_level_2"
            elif dhw >= 4:
                alert = "alert_level_1"
            elif hotspot > 0:
                alert = "watch"
            records.append(
                {
                    "reef_id": reef_id,
                    "date": date.isoformat(),
                    "sst_celsius": round(sst, 3),
                    "sst_anomaly_c": round(anomaly, 3),
                    "hotspot_c": round(hotspot, 3),
                    "degree_heating_weeks": round(dhw, 3),
                    "bleaching_alert_area": alert,
                }
            )
    logger.info("Generated %d days of NOAA-style data for %d reefs", days, len(reef_ids))
    return pd.DataFrame(records)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(settings.noaa_output))
    args = parser.parse_args()
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    df = generate_noaa_sample()
    df.to_csv(out, index=False)
    logger.info("Wrote NOAA-style sample data to %s", out)


if __name__ == "__main__":
    main()
