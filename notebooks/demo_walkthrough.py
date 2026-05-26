"""Generate the ReefTwin demo walkthrough notebook programmatically."""

import nbformat as nbf

nb = nbf.v4.new_notebook()
nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}

cells = []

def md(source):
    cells.append(nbf.v4.new_markdown_cell(source))

def code(source):
    cells.append(nbf.v4.new_code_cell(source))


# ============================================================
md("""# ReefTwin Demo Walkthrough

**Real-Time Digital Twin for Coral Reef Ecosystems**

This notebook demonstrates the full ReefTwin platform in ~10 minutes:
1. Data generation and feature engineering (Polars)
2. Model training (RandomForest + Physics-Informed ML)
3. Reef state prediction with uncertainty
4. Scenario simulation with AI interpretation
5. Anomaly detection + forecasting
6. RAG knowledge base search
7. Drift monitoring (Evidently AI)
8. Ecosystem graph + stress propagation
""")

# --- Section 1: Data Pipeline ---
md("## 1. Data Pipeline (Polars)")

code("""# Generate synthetic IoT sensor readings and NOAA heat-stress data
from pipelines.simulate_iot_stream import generate_readings
from pipelines.ingest_noaa_crw import generate_noaa_sample
from pipelines.build_features import build_features

iot_df = generate_readings(1000)
print(f"IoT readings: {len(iot_df)} rows")
iot_df.head()
""")

code("""noaa_df = generate_noaa_sample(30)
print(f"NOAA data: {len(noaa_df)} rows")
noaa_df.head()
""")

code("""# Feature engineering with Polars (fast!)
import tempfile, os
tmp = tempfile.mkdtemp()
iot_df.to_csv(f"{tmp}/iot.csv", index=False)
noaa_df.to_csv(f"{tmp}/noaa.csv", index=False)

features = build_features(f"{tmp}/iot.csv", f"{tmp}/noaa.csv")
print(f"Features: {features.shape}")
features.head()
""")

# --- Section 2: Model Training ---
md("## 2. Model Training")

code("""# Train the RandomForest bleaching risk classifier
from models.bleaching_risk.train import train_model

features.to_csv(f"{tmp}/features.csv", index=False)
rf_metrics = train_model(f"{tmp}/features.csv", f"{tmp}/rf_model.joblib")
print("RandomForest metrics:", rf_metrics)
""")

code("""# Train the Physics-Informed ML hybrid model
from models.reef_dynamics.hybrid_predictor import train_hybrid_model

hybrid_metrics = train_hybrid_model(f"{tmp}/features.csv", f"{tmp}/hybrid_model.joblib")
print("PIML Hybrid metrics:", hybrid_metrics)
""")

# --- Section 3: Prediction with Uncertainty ---
md("## 3. Prediction with Uncertainty Quantification")

code("""from models.predictor import get_predictor
from models.uncertainty import ConformalPredictor
import numpy as np

# Use the strategy pattern to select a model
predictor = get_predictor("random_forest", rf_model_path=f"{tmp}/rf_model.joblib")
row = features.iloc[0].to_dict()
result = predictor.predict(row)
print(f"Strategy: {result.model_strategy}")
print(f"Risk score: {result.bleaching_risk_score}")
print(f"Category: {result.risk_category}")
""")

code("""# Add uncertainty via conformal prediction
from models.bleaching_risk.inference import predict_risk

# Calibrate on test set
scores = []
actuals = []
for _, r in features.iterrows():
    pred = predict_risk(f"{tmp}/rf_model.joblib", r.to_dict())
    scores.append(pred["bleaching_risk_score"])
    actuals.append(float(r["bleaching_label"]))

cp = ConformalPredictor(confidence=0.90)
cp.calibrate(np.array(actuals), np.array(scores))

# Predict with interval
result_with_unc = cp.predict(scores[0])
print(f"Point estimate: {result_with_unc.point_estimate}")
print(f"90% interval: [{result_with_unc.lower_bound}, {result_with_unc.upper_bound}]")
print(f"Interval width: {result_with_unc.interval_width}")
""")

# --- Section 4: Scenario Simulation ---
md("## 4. Scenario Simulation")

code("""# Simulate: what if temperature rises 2°C for 30 days?
from infrastructure.settings import settings

base_risk = result.bleaching_risk_score
temp_delta = 2.0
duration = 30

temp_pressure = max(0, temp_delta) * settings.sim_temperature_weight
duration_pressure = min(duration / 90, 1.0) * settings.sim_duration_weight
projected = min(1.0, base_risk + temp_pressure + duration_pressure)

print(f"Baseline risk:  {base_risk:.3f}")
print(f"Projected risk: {projected:.3f} (+{temp_delta}°C for {duration} days)")
print(f"Risk change:    +{(projected - base_risk):.3f}")
""")

code("""# Physics-based simulation using the ODE model
from models.reef_dynamics.physics import simulate_reef_stress
import numpy as np

# 12-week SST series: stable then heat wave
sst = np.concatenate([np.full(6, 28.0), np.full(6, 31.0)])
result = simulate_reef_stress(sst)

print("Week  | DHW    | Stress | Risk")
print("-" * 40)
for i in [0, 3, 6, 9, 11]:
    print(f"  {i+1:2d}   | {result['dhw'][i]:5.2f}  | {result['stress'][i]:5.3f}  | {result['bleaching_risk'][i]:5.3f}")
""")

# --- Section 5: Anomaly Detection + Forecasting ---
md("## 5. Anomaly Detection + Forecasting")

code("""# Train anomaly detector
from models.anomaly_detection.detector import train_anomaly_detector, detect_anomaly

ad_metrics = train_anomaly_detector(f"{tmp}/features.csv", f"{tmp}/anomaly.joblib", contamination=0.1)
print("Anomaly detector:", ad_metrics)

# Test normal vs extreme reading
normal = {"water_temperature_c": 28.3, "ph": 8.05, "salinity_psu": 35.1,
          "turbidity_ntu": 0.8, "dissolved_oxygen_mg_l": 6.5}
extreme = {"water_temperature_c": 38.0, "ph": 6.0, "salinity_psu": 50.0,
           "turbidity_ntu": 20.0, "dissolved_oxygen_mg_l": 1.0}

print(f"Normal reading:  anomaly={detect_anomaly(f'{tmp}/anomaly.joblib', normal).is_anomaly}")
print(f"Extreme reading: anomaly={detect_anomaly(f'{tmp}/anomaly.joblib', extreme).is_anomaly}")
""")

code("""# Forecast SST for next 7 days
from models.forecasting.forecaster import forecast_parameter
import pandas as pd

sst_series = features[features["reef_id"] == "gbr_heron_reef"]["water_temperature_c"].dropna()
fcast = forecast_parameter(sst_series, horizon=7, backend="holtwinters")

print(f"Last observed: {float(sst_series.iloc[-1]):.2f}°C")
print(f"Trend: {fcast['trend']}")
print(f"7-day forecast: {[f'{v:.2f}' for v in fcast['forecast']]}")
print(f"Backend: {fcast['backend']}")
""")

# --- Section 6: RAG Knowledge Base ---
md("## 6. Reef Knowledge RAG")

code("""# Search the reef science knowledge base
from infrastructure.genai.rag import HybridRAGPipeline

pipeline = HybridRAGPipeline()
result = pipeline.query("What causes coral bleaching and what are degree heating weeks?")

print("Answer:", result.answer[:300], "...")
print(f"\\nSources: {len(result.sources)} retrieved")
for s in result.sources:
    print(f"  [{s['metadata'].get('source')}] {s['metadata'].get('topic')}: {s['content'][:80]}...")
""")

# --- Section 7: Drift Monitoring ---
md("## 7. Drift Monitoring (Evidently AI)")

code("""# Detect drift between reference and current data
from infrastructure.mlops.evidently_drift import run_data_drift_report
import numpy as np

rng = np.random.default_rng(42)
reference = pd.DataFrame({
    "water_temperature_c": rng.normal(28.3, 0.5, 200),
    "ph": rng.normal(8.1, 0.05, 200),
})
# Simulate heat wave (shifted distribution)
current = pd.DataFrame({
    "water_temperature_c": rng.normal(31.0, 0.8, 200),  # +2.7°C shift!
    "ph": rng.normal(8.1, 0.05, 200),                    # no change
})

drift = run_data_drift_report(reference, current)
print(f"Drift detected: {drift.is_drifted}")
print(f"Drifted features: {drift.n_drifted_features}/{drift.n_total_features}")
for feat, detail in drift.feature_details.items():
    print(f"  {feat}: drifted={detail['drifted']}, method={detail['stattest']}, score={detail['drift_score']:.4f}")
""")

# --- Section 8: Ecosystem Graph ---
md("## 8. Ecosystem Graph + Stress Propagation")

code("""from models.ecosystem_graph import build_reef_ecosystem_graph, simulate_stress_propagation, get_ecosystem_summary

G = build_reef_ecosystem_graph()
summary = get_ecosystem_summary(G)
print(f"Ecosystem: {summary['nodes']} nodes, {summary['edges']} edges")
print(f"Node types: {summary['node_types']}")
print(f"Most connected: {summary['most_connected'][:3]}")

# Simulate thermal stress propagation
stress = simulate_stress_propagation(G, initial_stress={"water_temperature": 0.9}, propagation_steps=3)
print(f"\\nStress propagation from thermal event:")
for node, level in sorted(stress.items(), key=lambda x: -x[1])[:6]:
    print(f"  {node:25s} → {level:.3f}")
""")

# --- Section 9: Experiments ---
md("## 9. Experiment Results")

code("""from infrastructure.mlops.experiments import run_experiment_2_cost, run_experiment_3_reliability, generate_report

results = [run_experiment_2_cost(), run_experiment_3_reliability()]
print(generate_report(results))
""")

md("""## Summary

This walkthrough demonstrated:

| Component | What you saw |
|-----------|-------------|
| **Polars** | Feature engineering (IoT + NOAA → daily aggregates) |
| **RandomForest** | Bleaching risk classification with configurable thresholds |
| **PIML Hybrid** | Physics ODE + ML residual correction |
| **Conformal Prediction** | Calibrated 90% prediction intervals |
| **Physics ODE** | DHW accumulation + stress dynamics simulation |
| **Anomaly Detection** | Isolation Forest on sensor readings |
| **Forecasting** | Holt-Winters SST forecast with prediction intervals |
| **RAG** | Hybrid search (BM25 + dense + RRF) on reef science corpus |
| **Evidently AI** | Distribution drift detection with statistical tests |
| **Ecosystem Graph** | NetworkX stress propagation through reef food web |
| **Experiments** | Measurable cost reduction + reliability improvement |

All features are API-served via FastAPI, visualised in Streamlit + React dashboards, and tracked with MLflow.
""")

nb.cells = cells
nbf.write(nb, "notebooks/demo_walkthrough.ipynb")
print("Demo notebook written: notebooks/demo_walkthrough.ipynb")
