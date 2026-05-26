from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl

from infrastructure.logging import get_logger
from infrastructure.settings import settings

logger = get_logger("pipelines.build_features")


def build_features(iot_path: str, noaa_path: str) -> "pd.DataFrame":
    """Build reef feature table from IoT + NOAA data using Polars.

    Returns a pandas DataFrame for compatibility with downstream
    scikit-learn models. Internal processing uses Polars for speed.
    """
    iot = pl.read_csv(iot_path, try_parse_dates=True)
    noaa = pl.read_csv(noaa_path, try_parse_dates=True)

    # Extract date string for join key
    iot = iot.with_columns(
        pl.col("timestamp").cast(pl.String).str.slice(0, 10).alias("date")
    )
    noaa = noaa.with_columns(
        pl.col("date").cast(pl.String).str.slice(0, 10)
    )

    # Aggregate IoT readings to daily means per reef
    agg = (
        iot.group_by(["reef_id", "date"])
        .agg(
            pl.col("water_temperature_c").mean(),
            pl.col("ph").mean(),
            pl.col("salinity_psu").mean(),
            pl.col("turbidity_ntu").mean(),
            pl.col("dissolved_oxygen_mg_l").mean(),
        )
        .sort(["reef_id", "date"])
    )

    # Join with NOAA heat-stress data
    features = agg.join(noaa, on=["reef_id", "date"], how="left")

    # Forward-fill NOAA columns within each reef, then global median
    # (never fill with 0 — zeros are not valid proxies for environmental data)
    noaa_cols = ["sst_anomaly_c", "hotspot_c", "degree_heating_weeks"]
    features = features.with_columns(
        [pl.col(c).forward_fill().backward_fill().over("reef_id") for c in noaa_cols]
    )
    for col in noaa_cols:
        median_val = features[col].drop_nulls().median()
        if median_val is not None:
            features = features.with_columns(pl.col(col).fill_null(median_val))

    # 7-day temperature trend (rolling mean - lagged value)
    features = features.with_columns(
        (
            pl.col("water_temperature_c")
            .rolling_mean(window_size=7, min_samples=1)
            .over("reef_id")
            - pl.col("water_temperature_c")
            .shift(7)
            .over("reef_id")
            .fill_null(pl.col("water_temperature_c").mean().over("reef_id"))
        ).alias("temperature_trend_7d")
    )

    # Bleaching label based on configurable thresholds
    features = features.with_columns(
        (
            (pl.col("degree_heating_weeks") >= settings.bleaching_dhw_threshold)
            | (pl.col("water_temperature_c") >= settings.bleaching_temp_threshold)
            | (pl.col("hotspot_c") >= settings.bleaching_hotspot_threshold)
        )
        .cast(pl.Int32)
        .alias("bleaching_label")
    )

    # Convert to pandas for downstream scikit-learn compatibility
    pdf = features.to_pandas()
    logger.info("Built %d feature rows from %s + %s (Polars)", len(pdf), iot_path, noaa_path)
    return pdf


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iot", default=str(settings.iot_output))
    parser.add_argument("--noaa", default=str(settings.noaa_output))
    parser.add_argument("--output", default=str(settings.features_path))
    args = parser.parse_args()
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    df = build_features(args.iot, args.noaa)
    if out.suffix == ".parquet":
        df.to_parquet(out, index=False)
    else:
        df.to_csv(out, index=False)
    logger.info("Wrote feature table to %s (%d rows)", out, len(df))


if __name__ == "__main__":
    main()
