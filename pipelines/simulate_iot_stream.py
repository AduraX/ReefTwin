from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from infrastructure.logging import get_logger
from infrastructure.settings import settings

logger = get_logger("pipelines.simulate_iot_stream")


def generate_readings(rows: int = 5000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    reef_ids = settings.reef_ids
    now = datetime.now(timezone.utc).replace(microsecond=0)
    records = []
    for i in range(rows):
        reef_id = reef_ids[i % len(reef_ids)]
        ts = now - timedelta(minutes=rows - i)
        heat_wave = 1.8 if i > rows * 0.72 and reef_id == reef_ids[0] else 0.0
        temp = rng.normal(28.3, 0.45) + heat_wave
        ph = rng.normal(8.05, 0.06) - (0.07 if heat_wave else 0)
        turbidity = max(0.05, rng.normal(0.8, 0.22) + (0.45 if heat_wave else 0))
        records.append(
            {
                "reef_id": reef_id,
                "timestamp": ts.isoformat(),
                "water_temperature_c": round(temp, 3),
                "ph": round(ph, 3),
                "salinity_psu": round(rng.normal(35.1, 0.35), 3),
                "turbidity_ntu": round(turbidity, 3),
                "dissolved_oxygen_mg_l": round(rng.normal(6.5, 0.4) - heat_wave * 0.2, 3),
            }
        )
    logger.info("Generated %d IoT readings for %d reefs", rows, len(reef_ids))
    return pd.DataFrame(records)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(settings.iot_output))
    parser.add_argument("--rows", type=int, default=5000)
    args = parser.parse_args()
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    generate_readings(args.rows).to_csv(out, index=False)
    logger.info("Wrote %d IoT readings to %s", args.rows, out)


if __name__ == "__main__":
    main()
