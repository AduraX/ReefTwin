"""ReefTwin Interactive Dashboard.

Streamlit-based decision-support dashboard for reef managers.
Provides: reef state overview, bleaching risk map, scenario simulation,
stress breakdown, and knowledge base search.

Run: streamlit run services/dashboard/app.py
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

st.set_page_config(page_title="ReefTwin Dashboard", page_icon="🪸", layout="wide")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_reef_states() -> list[dict]:
    from infrastructure.db.factory import get_state_store
    return get_state_store().load_states()


def load_reef_configs() -> list[dict]:
    import yaml
    path = Path(__file__).resolve().parent.parent.parent / "configs" / "reefs.yml"
    if path.exists():
        with open(path) as f:
            return yaml.safe_load(f).get("reefs", [])
    return []


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

st.sidebar.title("ReefTwin")
st.sidebar.markdown("Real-Time Digital Twin for Coral Reef Ecosystems")
page = st.sidebar.radio(
    "Navigate",
    ["Overview", "Scenario Simulation", "Stress Analysis", "Knowledge Base", "Data Upload"],
)


# ---------------------------------------------------------------------------
# Page: Overview
# ---------------------------------------------------------------------------

if page == "Overview":
    st.title("Reef State Overview")

    states = load_reef_states()
    configs = load_reef_configs()

    if not states:
        st.warning("No reef state data. Run `make update-twin` first.")
        st.stop()

    # KPI cards
    cols = st.columns(len(states))
    for i, state in enumerate(states):
        with cols[i]:
            reef_name = state["reef_id"].replace("_", " ").title()
            risk = state.get("bleaching_risk_score", 0)
            status = state.get("ecosystem_status", "unknown")

            color = {"stable": "green", "watch": "orange", "stressed": "red", "critical": "darkred"}.get(status, "gray")

            st.metric(label=reef_name, value=f"{risk:.2%}", delta=status)
            st.markdown(f"**Temp:** {state.get('water_temperature_c', 'N/A')}°C")
            st.markdown(f"**DHW:** {state.get('degree_heating_weeks', 'N/A')}")
            st.markdown(f"**pH:** {state.get('ph', 'N/A')}")

    # Reef map
    st.subheader("Reef Locations")
    if configs:
        import plotly.graph_objects as go

        fig = go.Figure()
        for cfg in configs:
            # Find matching state
            state = next((s for s in states if s["reef_id"] == cfg["reef_id"]), {})
            risk = state.get("bleaching_risk_score", 0)

            risk_color = "green" if risk < 0.5 else "orange" if risk < 0.7 else "red" if risk < 0.85 else "darkred"

            fig.add_trace(go.Scattermap(
                lat=[cfg["latitude"]],
                lon=[cfg["longitude"]],
                mode="markers+text",
                marker=dict(size=20, color=risk_color),
                text=[cfg["name"]],
                textposition="top center",
                name=cfg["name"],
                hovertemplate=(
                    f"<b>{cfg['name']}</b><br>"
                    f"Risk: {risk:.2%}<br>"
                    f"Status: {state.get('ecosystem_status', 'N/A')}<br>"
                    f"Temp: {state.get('water_temperature_c', 'N/A')}°C"
                    "<extra></extra>"
                ),
            ))

        fig.update_layout(
            map=dict(style="open-street-map", center=dict(lat=-18, lon=150), zoom=4),
            height=500,
            margin=dict(l=0, r=0, t=0, b=0),
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)

    # Risk comparison bar chart
    st.subheader("Bleaching Risk Comparison")
    import plotly.express as px
    import pandas as pd

    risk_df = pd.DataFrame([
        {"reef": s["reef_id"].replace("_", " ").title(), "risk": s.get("bleaching_risk_score", 0)}
        for s in states
    ])
    fig_bar = px.bar(
        risk_df, x="reef", y="risk",
        color="risk",
        color_continuous_scale=["green", "orange", "red"],
        range_color=[0, 1],
        labels={"risk": "Bleaching Risk", "reef": "Reef"},
    )
    fig_bar.add_hline(y=0.5, line_dash="dash", annotation_text="Watch", line_color="orange")
    fig_bar.add_hline(y=0.85, line_dash="dash", annotation_text="Alert", line_color="red")
    fig_bar.update_layout(height=350)
    st.plotly_chart(fig_bar, use_container_width=True)


# ---------------------------------------------------------------------------
# Page: Scenario Simulation
# ---------------------------------------------------------------------------

elif page == "Scenario Simulation":
    st.title("Scenario Simulation")

    states = load_reef_states()
    if not states:
        st.warning("No reef state data. Run `make update-twin` first.")
        st.stop()

    reef_ids = [s["reef_id"] for s in states]

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Parameters")
        reef_id = st.selectbox("Reef", reef_ids, format_func=lambda x: x.replace("_", " ").title())
        temp_delta = st.slider("Temperature change (°C)", -2.0, 5.0, 1.5, 0.1)
        duration = st.slider("Duration (days)", 1, 180, 21)
        turbidity_delta = st.slider("Turbidity change (%)", -50.0, 200.0, 0.0, 5.0)
        ph_delta = st.slider("pH change", -0.5, 0.2, 0.0, 0.01)
        run_sim = st.button("Run Simulation", type="primary")

    with col2:
        if run_sim:
            from infrastructure.settings import settings

            base_state = next((s for s in states if s["reef_id"] == reef_id), None)
            if base_state:
                base_risk = float(base_state["bleaching_risk_score"])
                temp_pressure = max(0, temp_delta) * settings.sim_temperature_weight
                duration_pressure = min(duration / 90, 1.0) * settings.sim_duration_weight
                turb_pressure = max(0, turbidity_delta) / 100 * settings.sim_turbidity_weight
                acid_pressure = max(0, -ph_delta) * settings.sim_acidification_weight
                projected = min(1.0, base_risk + temp_pressure + duration_pressure + turb_pressure + acid_pressure)

                st.subheader("Results")
                c1, c2, c3 = st.columns(3)
                c1.metric("Baseline Risk", f"{base_risk:.2%}")
                c2.metric("Projected Risk", f"{projected:.2%}", delta=f"{(projected - base_risk):+.2%}")

                status = "stable"
                t = settings.risk_thresholds
                if projected >= t.alert: status = "critical"
                elif projected >= t.warning: status = "stressed"
                elif projected >= t.watch: status = "watch"
                c3.metric("Status", status)

                # Risk gauge
                import plotly.graph_objects as go
                fig = go.Figure(go.Indicator(
                    mode="gauge+number+delta",
                    value=projected,
                    delta={"reference": base_risk},
                    gauge={
                        "axis": {"range": [0, 1]},
                        "bar": {"color": "darkred" if projected >= 0.85 else "red" if projected >= 0.7 else "orange" if projected >= 0.5 else "green"},
                        "steps": [
                            {"range": [0, 0.5], "color": "lightgreen"},
                            {"range": [0.5, 0.7], "color": "lightyellow"},
                            {"range": [0.7, 0.85], "color": "lightsalmon"},
                            {"range": [0.85, 1], "color": "lightcoral"},
                        ],
                        "threshold": {"line": {"color": "black", "width": 3}, "thickness": 0.8, "value": base_risk},
                    },
                    title={"text": "Projected Bleaching Risk"},
                ))
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)

                # NL interpretation (mock if no API key)
                st.subheader("AI Interpretation")
                from infrastructure.genai.scenario_interpreter import interpret_simulation
                sim_result = {
                    "reef_id": reef_id,
                    "baseline_risk": base_risk,
                    "projected_bleaching_risk": projected,
                    "projected_ecosystem_status": status,
                    "scenario": {
                        "temperature_delta_c": temp_delta,
                        "duration_days": duration,
                        "turbidity_delta_pct": turbidity_delta,
                        "ph_delta": ph_delta,
                    },
                }
                interp = interpret_simulation(sim_result, base_state)
                st.markdown(f"**Summary:** {interp.summary}")
                st.markdown(f"**Risk Assessment:** {interp.risk_assessment}")
                st.markdown("**Recommendations:**")
                for rec in interp.recommendations:
                    st.markdown(f"- {rec}")


# ---------------------------------------------------------------------------
# Page: Stress Analysis
# ---------------------------------------------------------------------------

elif page == "Stress Analysis":
    st.title("Multi-Factor Stress Analysis")

    states = load_reef_states()
    if not states:
        st.warning("No reef state data.")
        st.stop()

    from models.stress_scoring import ReefStressModel
    import plotly.graph_objects as go

    model = ReefStressModel()

    for state in states:
        reef_name = state["reef_id"].replace("_", " ").title()
        breakdown = model.score(state)

        st.subheader(f"{reef_name} — Total Stress: {breakdown.total_score:.2%}")
        st.caption(f"Dominant stressor: **{breakdown.dominant_stressor}**")

        # Radar chart
        categories = ["Thermal", "Water Quality", "Biological", "Cumulative"]
        values = [
            breakdown.thermal_score,
            breakdown.water_quality_score,
            breakdown.biological_score,
            breakdown.cumulative_score,
        ]

        fig = go.Figure(go.Scatterpolar(
            r=values + [values[0]],
            theta=categories + [categories[0]],
            fill="toself",
            name=reef_name,
            line_color="red" if breakdown.total_score > 0.5 else "orange" if breakdown.total_score > 0.3 else "green",
        ))
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            height=350,
            margin=dict(l=60, r=60, t=30, b=30),
        )
        st.plotly_chart(fig, use_container_width=True)
        st.divider()


# ---------------------------------------------------------------------------
# Page: Knowledge Base
# ---------------------------------------------------------------------------

elif page == "Knowledge Base":
    st.title("Reef Knowledge Base")
    st.markdown("Search reef science literature using hybrid RAG (BM25 + dense + Reciprocal Rank Fusion).")

    query = st.text_input("Ask a question about coral reefs:", placeholder="What causes coral bleaching?")

    if query:
        from infrastructure.genai.rag import HybridRAGPipeline
        pipeline = HybridRAGPipeline()
        result = pipeline.query(query)

        st.subheader("Answer")
        st.markdown(result.answer)

        with st.expander(f"Sources ({len(result.sources)} retrieved)"):
            for src in result.sources:
                st.markdown(f"**[{src['metadata'].get('source', '?')}]** {src['metadata'].get('topic', '')}")
                st.caption(src["content"][:200] + "...")
                st.divider()

        st.caption(f"Model: {result.model} | Retrieval: {result.retrieval_method}")


# ---------------------------------------------------------------------------
# Page: Data Upload
# ---------------------------------------------------------------------------

elif page == "Data Upload":
    st.title("Upload Reef Datasets")
    st.markdown(
        "Upload **CSV**, **Parquet**, or **JSON** files to the **bronze** data layer. "
        "After uploading, re-run the pipeline: `make build-features && make train-model && make update-twin`"
    )
    st.info("For real-time streaming, push JSON events to `POST /ingest/stream` (Kafka/Redpanda).")

    dataset_type = st.selectbox("Dataset type", ["iot", "noaa"], format_func=lambda x: {
        "iot": "IoT Sensor Readings (iot_readings.csv)",
        "noaa": "NOAA CRW Satellite Data (noaa_crw_sample.csv)",
    }[x])

    with st.expander("Expected CSV schema"):
        if dataset_type == "iot":
            st.markdown("""
| Column | Type | Example |
|--------|------|---------|
| `reef_id` | string | `gbr_heron_reef` |
| `timestamp` | ISO datetime | `2026-05-07T10:30:00` |
| `water_temperature_c` | float | `28.3` |
| `ph` | float | `8.05` |
| `salinity_psu` | float | `35.1` |
| `turbidity_ntu` | float | `0.8` |
| `dissolved_oxygen_mg_l` | float | `6.5` |
""")
        else:
            st.markdown("""
| Column | Type | Example |
|--------|------|---------|
| `reef_id` | string | `gbr_heron_reef` |
| `date` | date | `2026-05-07` |
| `sst_celsius` | float | `28.9` |
| `sst_anomaly_c` | float | `0.5` |
| `hotspot_c` | float | `0.3` |
| `degree_heating_weeks` | float | `2.1` |
| `bleaching_alert_area` | int | `1` |
""")

    uploaded = st.file_uploader("Choose a CSV, Parquet, or JSON file", type=["csv", "parquet", "json"])

    if uploaded is not None:
        import pandas as pd
        try:
            if uploaded.name.endswith(".parquet"):
                df = pd.read_parquet(uploaded)
            elif uploaded.name.endswith(".json"):
                import json as _json
                records = _json.loads(uploaded.read())
                if not isinstance(records, list):
                    st.error("JSON must be an array of objects")
                    st.stop()
                df = pd.DataFrame(records)
            else:
                df = pd.read_csv(uploaded)
            st.subheader("Preview")
            st.dataframe(df.head(20), use_container_width=True)
            st.caption(f"{len(df)} rows, {len(df.columns)} columns")

            # Validate columns
            expected_iot = {"reef_id", "timestamp", "water_temperature_c", "ph", "salinity_psu", "turbidity_ntu", "dissolved_oxygen_mg_l"}
            expected_noaa = {"reef_id", "date", "sst_celsius", "sst_anomaly_c", "hotspot_c", "degree_heating_weeks", "bleaching_alert_area"}
            expected = expected_iot if dataset_type == "iot" else expected_noaa
            actual = set(df.columns)
            missing = expected - actual

            if missing:
                st.error(f"Missing required columns: {sorted(missing)}")
            else:
                st.success("Schema valid")

                if st.button("Upload to bronze layer", type="primary"):
                    target_name = "iot_readings.csv" if dataset_type == "iot" else "noaa_crw_sample.csv"

                    from infrastructure.settings import settings
                    if settings.state_store_backend == "s3":
                        from infrastructure.db.s3_store import S3DataStore
                        store = S3DataStore()
                        import io
                        buf = io.BytesIO()
                        df.to_csv(buf, index=False)
                        store.put_bytes(f"bronze/{target_name}", buf.getvalue())
                    else:
                        bronze_path = Path(__file__).resolve().parent.parent.parent / "data" / "bronze" / target_name
                        bronze_path.parent.mkdir(parents=True, exist_ok=True)
                        df.to_csv(bronze_path, index=False)

                    st.success(f"Uploaded {len(df)} rows to `bronze/{target_name}`")
                    st.info("Next: run `make build-features && make train-model && make update-twin` to update the twin state.")
        except Exception as e:
            st.error(f"Failed to read CSV: {e}")
