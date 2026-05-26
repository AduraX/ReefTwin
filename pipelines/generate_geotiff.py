"""Generate GeoTIFF raster files from reef feature data.

Produces spatial heatmaps for SST, bleaching risk, and DHW that can be
overlaid on dashboard maps. Uses inverse-distance weighted interpolation
to create continuous surfaces from discrete reef point data.

When rasterio is not installed, falls back to generating a numpy array
that can be served as a PNG tile.

Usage:
    python -m pipelines.generate_geotiff --features data/gold/reef_features.parquet
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from infrastructure.logging import get_logger
from infrastructure.settings import read_df, settings

logger = get_logger("pipelines.generate_geotiff")

# Bounding box for GBR + Coral Sea region
BBOX = {
    "west": 143.0,
    "east": 155.0,
    "south": -25.0,
    "north": -13.0,
}
RESOLUTION = 0.05  # degrees per pixel (~5.5km)


def _load_reef_coords() -> dict[str, tuple[float, float]]:
    """Load reef_id -> (lat, lon) from configs/reefs.yml."""
    import yaml
    path = Path(__file__).resolve().parent.parent / "configs" / "reefs.yml"
    if not path.exists():
        return {}
    with open(path) as f:
        reefs = yaml.safe_load(f).get("reefs", [])
    return {r["reef_id"]: (r["latitude"], r["longitude"]) for r in reefs}


def _idw_interpolate(
    points: list[tuple[float, float, float]],
    grid_lat: np.ndarray,
    grid_lon: np.ndarray,
    power: float = 2.0,
) -> np.ndarray:
    """Inverse Distance Weighting interpolation on a lat/lon grid."""
    result = np.full((len(grid_lat), len(grid_lon)), np.nan)
    if not points:
        return result

    for i, lat in enumerate(grid_lat):
        for j, lon in enumerate(grid_lon):
            weights = []
            values = []
            for plat, plon, val in points:
                dist = np.sqrt((lat - plat) ** 2 + (lon - plon) ** 2)
                if dist < 1e-10:
                    result[i, j] = val
                    break
                weights.append(1.0 / dist ** power)
                values.append(val)
            else:
                total_w = sum(weights)
                if total_w > 0:
                    result[i, j] = sum(w * v for w, v in zip(weights, values)) / total_w

    return result


def generate_geotiff(
    features_path: str,
    output_dir: str = "data/gold/geotiff",
    layers: list[str] | None = None,
) -> dict[str, str]:
    """Generate GeoTIFF files for spatial reef data layers.

    Args:
        features_path: Path to the features Parquet/CSV file.
        output_dir: Directory to write GeoTIFF files.
        layers: Which layers to generate. Defaults to all.

    Returns:
        Dict mapping layer name to output file path.
    """
    df = read_df(features_path)
    coords = _load_reef_coords()

    if layers is None:
        layers = ["sst", "bleaching_risk", "dhw"]

    # Get latest values per reef
    latest = df.sort_values("date").groupby("reef_id", as_index=False).tail(1)

    layer_columns = {
        "sst": "water_temperature_c",
        "bleaching_risk": "bleaching_label",
        "dhw": "degree_heating_weeks",
    }

    grid_lat = np.arange(BBOX["north"], BBOX["south"], -RESOLUTION)
    grid_lon = np.arange(BBOX["west"], BBOX["east"], RESOLUTION)

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, str] = {}

    for layer in layers:
        col = layer_columns.get(layer)
        if col is None or col not in latest.columns:
            logger.warning("Skipping layer '%s': column '%s' not in features", layer, col)
            continue

        points = []
        for _, row in latest.iterrows():
            rid = row["reef_id"]
            if rid in coords and not pd.isna(row[col]):
                lat, lon = coords[rid]
                points.append((lat, lon, float(row[col])))

        if not points:
            logger.warning("No data points for layer '%s'", layer)
            continue

        grid = _idw_interpolate(points, grid_lat, grid_lon)

        out_path = out_dir / f"{layer}_latest.tif"

        try:
            import rasterio
            from rasterio.transform import from_bounds

            transform = from_bounds(
                BBOX["west"], BBOX["south"], BBOX["east"], BBOX["north"],
                len(grid_lon), len(grid_lat),
            )
            with rasterio.open(
                str(out_path),
                "w",
                driver="GTiff",
                height=len(grid_lat),
                width=len(grid_lon),
                count=1,
                dtype="float32",
                crs="EPSG:4326",
                transform=transform,
                nodata=np.nan,
            ) as dst:
                dst.write(grid.astype(np.float32), 1)

            logger.info("Wrote GeoTIFF: %s (%dx%d)", out_path, len(grid_lat), len(grid_lon))
        except ImportError:
            # Fallback: save as raw numpy for tile serving without rasterio
            np_path = out_dir / f"{layer}_latest.npy"
            np.save(str(np_path), grid.astype(np.float32))
            out_path = np_path
            logger.info("Wrote numpy grid (rasterio not installed): %s", np_path)

        outputs[layer] = str(out_path)

    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate GeoTIFF heatmaps from reef features")
    parser.add_argument("--features", default=str(settings.features_path))
    parser.add_argument("--output-dir", default="data/gold/geotiff")
    parser.add_argument("--layers", nargs="*", default=None)
    args = parser.parse_args()
    outputs = generate_geotiff(args.features, args.output_dir, args.layers)
    for layer, path in outputs.items():
        print(f"  {layer}: {path}")


if __name__ == "__main__":
    main()
