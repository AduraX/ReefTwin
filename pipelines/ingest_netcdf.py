"""Ingest NOAA CRW satellite data from NetCDF (.nc) files.

Supports local NetCDF files and remote OPeNDAP URLs. Extracts SST, SST anomaly,
HotSpot, DHW, and Bleaching Alert Area for configured reef locations.

Uses xarray for multi-dimensional slicing and nearest-neighbor coordinate lookup.

Usage:
    python -m pipelines.ingest_netcdf --input data.nc --output data/bronze/noaa_crw_sample.csv
    python -m pipelines.ingest_netcdf --input https://coastwatch.pfeg.noaa.gov/erddap/griddap/NOAA_DHW
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd

from infrastructure.logging import get_logger
from infrastructure.settings import settings

logger = get_logger("pipelines.ingest_netcdf")

# Variable name mappings: NetCDF variable -> ReefTwin column
_VAR_MAPPING = {
    "analysed_sst": "sst_celsius",
    "CRW_SST": "sst_celsius",
    "sea_surface_temperature": "sst_celsius",
    "sst": "sst_celsius",
    "CRW_SSTANOMALY": "sst_anomaly_c",
    "sst_anomaly": "sst_anomaly_c",
    "CRW_HOTSPOT": "hotspot_c",
    "hotspot": "hotspot_c",
    "CRW_DHW": "degree_heating_weeks",
    "degree_heating_week": "degree_heating_weeks",
    "CRW_BAA": "bleaching_alert_area",
    "bleaching_alert_area": "bleaching_alert_area",
}

# Reef locations from config
_REEF_LOCATIONS: list[dict[str, Any]] = []


def _get_reef_locations() -> list[dict[str, Any]]:
    global _REEF_LOCATIONS
    if _REEF_LOCATIONS:
        return _REEF_LOCATIONS

    import yaml
    path = Path(__file__).resolve().parent.parent / "configs" / "reefs.yml"
    if path.exists():
        with open(path) as f:
            reefs = yaml.safe_load(f).get("reefs", [])
        _REEF_LOCATIONS = reefs
    return _REEF_LOCATIONS


def ingest_netcdf(input_path: str) -> pd.DataFrame:
    """Read a NetCDF file and extract reef data for configured locations.

    Args:
        input_path: Local .nc file path or OPeNDAP URL.

    Returns:
        DataFrame with columns: reef_id, date, sst_celsius, sst_anomaly_c,
        hotspot_c, degree_heating_weeks, bleaching_alert_area.
    """
    try:
        import xarray as xr
    except ImportError:
        raise ImportError("Install xarray + netCDF4: pip install 'reeftwin[netcdf]'")

    logger.info("Opening NetCDF: %s", input_path)
    ds = xr.open_dataset(input_path)

    # Identify available variables
    available = {}
    for nc_var, rt_col in _VAR_MAPPING.items():
        if nc_var in ds.data_vars and rt_col not in available:
            available[rt_col] = nc_var
    logger.info("Found variables: %s", {v: k for k, v in available.items()})

    # Identify coordinate names
    lat_name = next((c for c in ds.coords if c.lower() in ("lat", "latitude")), None)
    lon_name = next((c for c in ds.coords if c.lower() in ("lon", "longitude")), None)
    time_name = next((c for c in ds.coords if c.lower() in ("time", "date")), None)

    if not lat_name or not lon_name:
        raise ValueError(f"Cannot find lat/lon coordinates in dataset. Available: {list(ds.coords)}")

    reefs = _get_reef_locations()
    if not reefs:
        raise ValueError("No reef locations configured in configs/reefs.yml")

    all_rows = []
    for reef in reefs:
        reef_id = reef["reef_id"]
        lat = reef["latitude"]
        lon = reef["longitude"]

        # Select nearest grid point
        point = ds.sel({lat_name: lat, lon_name: lon}, method="nearest")

        if time_name and time_name in point.dims:
            times = pd.to_datetime(point[time_name].values)
        else:
            times = [pd.Timestamp.now().normalize()]

        for i, t in enumerate(times):
            row: dict[str, Any] = {
                "reef_id": reef_id,
                "date": t.strftime("%Y-%m-%d"),
            }
            for rt_col, nc_var in available.items():
                if time_name and time_name in point[nc_var].dims:
                    val = float(point[nc_var].isel({time_name: i}).values)
                else:
                    val = float(point[nc_var].values)
                # Convert Kelvin to Celsius if needed (SST > 200 implies Kelvin)
                if rt_col == "sst_celsius" and val > 200:
                    val = val - 273.15
                row[rt_col] = round(val, 3)

            # Fill missing columns with defaults
            row.setdefault("sst_anomaly_c", 0.0)
            row.setdefault("hotspot_c", 0.0)
            row.setdefault("degree_heating_weeks", 0.0)
            row.setdefault("bleaching_alert_area", "normal")

            all_rows.append(row)

    ds.close()
    df = pd.DataFrame(all_rows)
    logger.info("Extracted %d records for %d reefs from NetCDF", len(df), len(reefs))
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest NOAA CRW data from NetCDF")
    parser.add_argument("--input", required=True, help="NetCDF file path or OPeNDAP URL")
    parser.add_argument("--output", default=str(settings.noaa_output))
    args = parser.parse_args()

    df = ingest_netcdf(args.input)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    logger.info("Wrote %d rows to %s", len(df), out)


if __name__ == "__main__":
    main()
