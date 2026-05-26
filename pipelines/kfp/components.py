"""Kubeflow Pipeline v2 components for ReefTwin.

Uses YAML-based component definitions (compatible with all KFP v2 versions).
Compile with: python -m pipelines.kfp.pipeline
"""

from kfp import dsl
from kfp.dsl import Dataset, Input, Model, Output


@dsl.component(
    base_image="python:3.12-slim",
    packages_to_install=["pandas", "numpy", "polars"],
)
def generate_iot_data(rows: int, iot_dataset: Output[Dataset]):
    import numpy as np
    import pandas as pd
    from datetime import datetime, timedelta, timezone
    reef_ids = ["gbr_heron_reef", "gbr_lizard_island", "coral_sea_reef"]
    rng = np.random.default_rng(42)
    now = datetime.now(timezone.utc).replace(microsecond=0)
    records = []
    for i in range(rows):
        rid = reef_ids[i % 3]
        hw = 1.8 if i > rows * 0.72 and rid == reef_ids[0] else 0.0
        records.append({"reef_id": rid, "timestamp": (now - timedelta(minutes=rows-i)).isoformat(),
            "water_temperature_c": round(float(rng.normal(28.3,0.45)+hw),3),
            "ph": round(float(rng.normal(8.05,0.06)-(0.07 if hw else 0)),3),
            "salinity_psu": round(float(rng.normal(35.1,0.35)),3),
            "turbidity_ntu": round(max(0.05,float(rng.normal(0.8,0.22)+(0.45 if hw else 0))),3),
            "dissolved_oxygen_mg_l": round(float(rng.normal(6.5,0.4)-hw*0.2),3)})
    pd.DataFrame(records).to_csv(iot_dataset.path, index=False)


@dsl.component(
    base_image="python:3.12-slim",
    packages_to_install=["pandas", "numpy"],
)
def generate_noaa_data(days: int, noaa_dataset: Output[Dataset]):
    import numpy as np
    import pandas as pd
    from datetime import datetime, timedelta, timezone
    reef_ids = ["gbr_heron_reef", "gbr_lizard_island", "coral_sea_reef"]
    rng = np.random.default_rng(7)
    today = datetime.now(timezone.utc).date()
    records = []
    for rid in reef_ids:
        hb = 1.2 if rid == reef_ids[0] else 0.3
        for d in range(days):
            date = today - timedelta(days=days-d-1)
            sst = rng.normal(28.5,0.5)+(hb if d > days*0.65 else 0)
            a = sst-28.2; h = max(0.0,a-0.7); dhw = max(0.0,h*min(8,d/7))
            al = "alert_level_2" if dhw>=8 else "alert_level_1" if dhw>=4 else "watch" if h>0 else "normal"
            records.append({"reef_id":rid,"date":date.isoformat(),"sst_celsius":round(sst,3),
                "sst_anomaly_c":round(a,3),"hotspot_c":round(h,3),
                "degree_heating_weeks":round(dhw,3),"bleaching_alert_area":al})
    pd.DataFrame(records).to_csv(noaa_dataset.path, index=False)


@dsl.component(
    base_image="python:3.12-slim",
    packages_to_install=["pandas", "numpy", "polars", "pyarrow"],
)
def build_features_component(
    iot_dataset: Input[Dataset],
    noaa_dataset: Input[Dataset],
    features_dataset: Output[Dataset],
):
    import polars as pl
    iot = pl.read_csv(iot_dataset.path, try_parse_dates=True)
    noaa = pl.read_csv(noaa_dataset.path, try_parse_dates=True)
    iot = iot.with_columns(pl.col("timestamp").cast(pl.String).str.slice(0,10).alias("date"))
    noaa = noaa.with_columns(pl.col("date").cast(pl.String).str.slice(0,10))
    agg = iot.group_by(["reef_id","date"]).agg(
        pl.col("water_temperature_c").mean(),pl.col("ph").mean(),
        pl.col("salinity_psu").mean(),pl.col("turbidity_ntu").mean(),
        pl.col("dissolved_oxygen_mg_l").mean()).sort(["reef_id","date"])
    f = agg.join(noaa, on=["reef_id","date"], how="left")
    for c in ["sst_anomaly_c","hotspot_c","degree_heating_weeks"]:
        f = f.with_columns(pl.col(c).forward_fill().backward_fill().over("reef_id"))
        m = f[c].drop_nulls().median()
        if m is not None: f = f.with_columns(pl.col(c).fill_null(m))
    f = f.with_columns((pl.col("water_temperature_c").rolling_mean(window_size=7,min_samples=1).over("reef_id")
        -pl.col("water_temperature_c").shift(7).over("reef_id").fill_null(
        pl.col("water_temperature_c").mean().over("reef_id"))).alias("temperature_trend_7d"))
    f = f.with_columns(((pl.col("degree_heating_weeks")>=4)|(pl.col("water_temperature_c")>=30)
        |(pl.col("hotspot_c")>=1)).cast(pl.Int32).alias("bleaching_label"))
    f.to_pandas().to_parquet(features_dataset.path, index=False)


@dsl.component(
    base_image="python:3.12-slim",
    packages_to_install=["pandas", "numpy", "scikit-learn", "joblib", "pyarrow"],
)
def train_model_component(
    features_dataset: Input[Dataset],
    trained_model: Output[Model],
):
    import pandas as pd
    import joblib
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    FEATURES = ["water_temperature_c","ph","salinity_psu","turbidity_ntu",
        "dissolved_oxygen_mg_l","sst_anomaly_c","hotspot_c","degree_heating_weeks","temperature_trend_7d"]
    df = pd.read_parquet(features_dataset.path)
    X = df[FEATURES].ffill().fillna(df[FEATURES].median())
    y = df["bleaching_label"].astype(int)
    strat = y if y.nunique()>1 and y.value_counts().min()>1 else None
    Xtr,Xte,ytr,yte = train_test_split(X,y,test_size=0.25,random_state=42,stratify=strat)
    model = Pipeline([("scaler",StandardScaler()),("clf",RandomForestClassifier(n_estimators=120,random_state=42,class_weight="balanced"))])
    model.fit(Xtr,ytr)
    joblib.dump({"model":model,"features":FEATURES,"metrics":{}}, trained_model.path)
