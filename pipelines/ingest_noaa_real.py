"""Real NOAA Coral Reef Watch API integration.

Fetches actual SST, HotSpot, DHW, and Bleaching Alert data from
NOAA CRW's public data products (no API key required).

Data source: NOAA Coral Reef Watch Daily 5km Satellite Products
URL: https://coralreefwatch.noaa.gov/product/5km/index_5km_sst.php

Falls back to synthetic data if network is unavailable.
"""

from __future__ import annotations

import argparse
import io
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from infrastructure.logging import get_logger
from infrastructure.settings import settings

logger = get_logger("pipelines.ingest_noaa_real")

# NOAA CRW virtual station coordinates for our reefs
REEF_STATIONS = {
    "gbr_heron_reef": {"lat": -23.442, "lon": 151.914, "name": "Heron Reef"},
    "gbr_lizard_island": {"lat": -14.668, "lon": 145.459, "name": "Lizard Island"},
    "coral_sea_reef": {"lat": -18.0, "lon": 152.0, "name": "Coral Sea"},
}

# NOAA CRW time series data endpoint (CSV format)
NOAA_CRW_BASE_URL = "https://coralreefwatch.noaa.gov/product/vs/data"


def fetch_noaa_timeseries(
    reef_id: str,
    days: int = 60,
) -> pd.DataFrame | None:
    """Fetch real NOAA CRW time-series data for a reef location.

    Uses NOAA's virtual station CSV endpoint. Returns None on failure.
    """
    import urllib.request
    import urllib.error

    station = REEF_STATIONS.get(reef_id)
    if not station:
        logger.warning("No NOAA station mapping for %s", reef_id)
        return None

    # NOAA CRW provides data via virtual stations nearest to coordinates
    # Fallback: use the global 5km SST anomaly product
    lat, lon = station["lat"], station["lon"]

    # Try NOAA CRW REST API for point data
    end_date = datetime.now(timezone.utc).date()
    start_date = end_date - timedelta(days=days)

    url = (
        f"https://coastwatch.pfeg.noaa.gov/erddap/griddap/NOAA_DHW.csv"
        f"?CRW_SST,CRW_SSTANOMALY,CRW_HOTSPOT,CRW_DHW,CRW_BAA"
        f"[({start_date}T12:00:00Z):1:({end_date}T12:00:00Z)]"
        f"[({lat}):1:({lat})]"
        f"[({lon}):1:({lon})]"
    )

    try:
        logger.info("Fetching NOAA CRW data for %s from CoastWatch ERDDAP...", reef_id)
        req = urllib.request.Request(url, headers={"User-Agent": "ReefTwin/1.0"})
        with urllib.request.urlopen(req, timeout=30) as response:
            csv_data = response.read().decode("utf-8")

        # Parse ERDDAP CSV (skip units row)
        df = pd.read_csv(io.StringIO(csv_data), skiprows=[1])

        if df.empty:
            logger.warning("NOAA returned empty data for %s", reef_id)
            return None

        # Rename columns to our schema
        col_map = {
            "time": "date",
            "CRW_SST": "sst_celsius",
            "CRW_SSTANOMALY": "sst_anomaly_c",
            "CRW_HOTSPOT": "hotspot_c",
            "CRW_DHW": "degree_heating_weeks",
            "CRW_BAA": "bleaching_alert_area",
        }
        df = df.rename(columns=col_map)
        df["reef_id"] = reef_id
        df["date"] = pd.to_datetime(df["date"]).dt.date.astype(str)

        # Map BAA numeric codes to labels
        baa_map = {0: "normal", 1: "watch", 2: "warning", 3: "alert_level_1", 4: "alert_level_2"}
        if "bleaching_alert_area" in df.columns:
            df["bleaching_alert_area"] = df["bleaching_alert_area"].map(
                lambda x: baa_map.get(int(x), "normal") if pd.notna(x) else "normal"
            )

        keep_cols = ["reef_id", "date", "sst_celsius", "sst_anomaly_c", "hotspot_c",
                     "degree_heating_weeks", "bleaching_alert_area"]
        df = df[[c for c in keep_cols if c in df.columns]]

        logger.info("Fetched %d rows of real NOAA data for %s", len(df), reef_id)
        return df

    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as e:
        logger.warning("NOAA API request failed for %s: %s", reef_id, e)
        return None
    except Exception as e:
        logger.warning("Failed to parse NOAA data for %s: %s", reef_id, e)
        return None


def ingest_noaa(
    days: int = 60,
    output: str | None = None,
    fallback_to_synthetic: bool = True,
) -> pd.DataFrame:
    """Ingest NOAA CRW data for all configured reefs.

    Tries real API first, falls back to synthetic if network unavailable.
    """
    frames = []

    for reef_id in settings.reef_ids:
        real_data = fetch_noaa_timeseries(reef_id, days=days)
        if real_data is not None and not real_data.empty:
            frames.append(real_data)
        elif fallback_to_synthetic:
            logger.info("Using synthetic data for %s (real API unavailable)", reef_id)
            from pipelines.ingest_noaa_crw import generate_noaa_sample
            synthetic = generate_noaa_sample(days=days)
            frames.append(synthetic[synthetic["reef_id"] == reef_id])

    if not frames:
        logger.warning("No NOAA data available — generating full synthetic dataset")
        from pipelines.ingest_noaa_crw import generate_noaa_sample
        return generate_noaa_sample(days=days)

    df = pd.concat(frames, ignore_index=True)

    if output:
        out = Path(output)
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out, index=False)
        logger.info("Wrote NOAA data to %s (%d rows, %d reefs)", out, len(df), df["reef_id"].nunique())

    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest real NOAA CRW data (with synthetic fallback)")
    parser.add_argument("--output", default=str(settings.noaa_output))
    parser.add_argument("--days", type=int, default=60)
    parser.add_argument("--no-fallback", action="store_true", help="Don't fall back to synthetic data")
    args = parser.parse_args()
    ingest_noaa(days=args.days, output=args.output, fallback_to_synthetic=not args.no_fallback)


if __name__ == "__main__":
    main()
