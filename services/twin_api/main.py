from __future__ import annotations

import io
from pathlib import Path
from time import perf_counter
from typing import Any

from fastapi import Depends, FastAPI, File, HTTPException, Response, UploadFile
from pydantic import BaseModel, Field
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from starlette.middleware.cors import CORSMiddleware

from infrastructure.db.factory import get_state_store
from infrastructure.logging import get_logger
from infrastructure.security import (
    Permission,
    check_rate_limit,
    require_auth,
    require_permission,
    require_reef_access,
    validate_query_length,
)
from infrastructure.settings import settings

logger = get_logger("services.twin_api")

app = FastAPI(
    title="ReefTwin API",
    version="0.3.0",
    description="Real-time digital twin platform for coral reef ecosystems",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:3001"], allow_methods=["*"], allow_headers=["*"])

# --- Prometheus Metrics ---
REQUEST_LATENCY = Histogram("api_request_latency_seconds", "API request latency", ["endpoint"])
SIMULATION_REQUESTS = Counter("simulation_requests_total", "Total simulation requests")
PREDICTION_LATENCY = Histogram("bleaching_prediction_latency_seconds", "Bleaching model inference latency")
RAG_QUERIES = Counter("rag_queries_total", "Total RAG queries")
AGENT_QUERIES = Counter("agent_queries_total", "Total agent queries")
AGENT_TOOL_CALLS = Counter("agent_tool_calls_total", "Total agent tool invocations", ["tool_name"])
REEF_STATE_UPDATES = Counter("reef_state_updates_total", "Total reef state updates")
DRIFT_ALERTS = Counter("drift_alerts_total", "Drift alerts triggered", ["severity"])


class SimulationRequest(BaseModel):
    reef_id: str
    temperature_delta_c: float = Field(default=0.0)
    duration_days: int = Field(default=21, ge=1, le=365)
    turbidity_delta_pct: float = Field(default=0.0)
    ph_delta: float = Field(default=0.0)


class RAGQuery(BaseModel):
    question: str
    k: int = Field(default=3, ge=1, le=10)


class AgentQuery(BaseModel):
    query: str


class QueryRequest(BaseModel):
    query: str


# ==========================================================================
# PUBLIC ENDPOINTS (no auth)
# ==========================================================================

@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/ready")
def ready() -> dict[str, Any]:
    """Readiness probe — verifies state store is accessible."""
    try:
        store = get_state_store()
        states = store.load_states()
        return {"status": "ready", "reefs_loaded": len(states)}
    except Exception as e:
        return {"status": "not_ready", "error": str(e)}


@app.get("/metrics")
def metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/tiles/{layer}")
def get_tile(layer: str) -> Response:
    """Serve a GeoTIFF or numpy grid as a downloadable raster file.

    Layers: sst, bleaching_risk, dhw.
    """
    geotiff_dir = Path(settings.state_path).parent / "geotiff"
    tif_path = geotiff_dir / f"{layer}_latest.tif"
    npy_path = geotiff_dir / f"{layer}_latest.npy"

    if tif_path.exists():
        return Response(
            tif_path.read_bytes(),
            media_type="image/tiff",
            headers={"Content-Disposition": f"attachment; filename={tif_path.name}"},
        )
    elif npy_path.exists():
        return Response(
            npy_path.read_bytes(),
            media_type="application/octet-stream",
            headers={"Content-Disposition": f"attachment; filename={npy_path.name}"},
        )
    raise HTTPException(status_code=404, detail=f"Tile layer '{layer}' not found. Run: make generate-geotiff")


@app.get("/public/reefs")
def public_reefs() -> dict[str, Any]:
    """Public reef summaries — no authentication required."""
    states = get_state_store().load_states()
    return {
        "reefs": [
            {
                "reef_id": s["reef_id"],
                "ecosystem_status": s.get("ecosystem_status", "unknown"),
                "bleaching_risk_score": s.get("bleaching_risk_score"),
            }
            for s in states
        ],
    }


# ==========================================================================
# PROTECTED ENDPOINTS (auth + RBAC via REEFTWIN_AUTH_MODE)
# ==========================================================================

@app.get(
    "/reefs/{reef_id}/state",
    dependencies=[Depends(require_reef_access(Permission.VIEW_REEF_STATE))],
)
def get_reef_state(reef_id: str) -> dict[str, Any]:
    start = perf_counter()
    try:
        for state in get_state_store().load_states():
            if state["reef_id"] == reef_id:
                return state
        raise HTTPException(status_code=404, detail="Reef state not found")
    finally:
        REQUEST_LATENCY.labels(endpoint="get_reef_state").observe(perf_counter() - start)


@app.get(
    "/reefs",
    dependencies=[Depends(require_permission(Permission.VIEW_REEF_STATE))],
)
def list_reefs() -> dict[str, Any]:
    return {"reefs": get_state_store().load_states()}


_EXPECTED_IOT = {"reef_id", "timestamp", "water_temperature_c", "ph", "salinity_psu", "turbidity_ntu", "dissolved_oxygen_mg_l"}
_EXPECTED_NOAA = {"reef_id", "date", "sst_celsius", "sst_anomaly_c", "hotspot_c", "degree_heating_weeks", "bleaching_alert_area"}


def _validate_columns(actual: set[str], dataset_type: str) -> None:
    expected = _EXPECTED_IOT if dataset_type == "iot" else _EXPECTED_NOAA
    missing = expected - actual
    if missing:
        raise HTTPException(status_code=400, detail=f"Missing columns: {sorted(missing)}")


def _write_bronze(target_name: str, data: bytes) -> None:
    if settings.state_store_backend == "s3":
        from infrastructure.db.s3_store import S3DataStore
        S3DataStore().put_bytes(f"bronze/{target_name}", data)
    else:
        bronze_path = Path(settings.state_path).parent.parent / "bronze" / target_name
        bronze_path.parent.mkdir(parents=True, exist_ok=True)
        bronze_path.write_bytes(data)


@app.post(
    "/datasets/upload",
    dependencies=[Depends(require_permission(Permission.UPLOAD_DATASET))],
)
async def upload_dataset(
    dataset_type: str,
    file: UploadFile = File(...),
) -> dict[str, Any]:
    """Upload a CSV, Parquet, JSON, or NetCDF dataset to the bronze layer.

    Requires scientist or reef_admin role.

    Args:
        dataset_type: ``iot`` or ``noaa`` — determines target filename.
        file: CSV (.csv), Parquet (.parquet), JSON (.json), or NetCDF (.nc) file.
              JSON must be an array of objects. NetCDF is only valid for ``noaa`` type.
    """
    if dataset_type not in ("iot", "noaa"):
        raise HTTPException(status_code=400, detail="dataset_type must be 'iot' or 'noaa'")

    filename = file.filename or ""
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if ext not in ("csv", "parquet", "json", "nc"):
        raise HTTPException(status_code=400, detail="Accepted formats: .csv, .parquet, .json, .nc")
    if ext == "nc" and dataset_type != "noaa":
        raise HTTPException(status_code=400, detail="NetCDF files are only accepted for dataset_type='noaa'")

    contents = await file.read()
    if len(contents) > 50 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File too large (max 50 MB)")

    # Parse, validate columns, and convert to CSV bytes for bronze layer
    if ext == "nc":
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".nc", delete=False) as tmp:
            tmp.write(contents)
            tmp_path = tmp.name
        try:
            from pipelines.ingest_netcdf import ingest_netcdf
            df = ingest_netcdf(tmp_path)
        except ImportError:
            raise HTTPException(status_code=400, detail="NetCDF support requires: pip install 'reeftwin[netcdf]'")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to parse NetCDF: {e}")
        finally:
            Path(tmp_path).unlink(missing_ok=True)
        _validate_columns(set(df.columns), dataset_type)
        row_count = len(df)
        csv_buf = io.BytesIO()
        df.to_csv(csv_buf, index=False)
        store_bytes = csv_buf.getvalue()

    elif ext == "parquet":
        import pandas as pd
        try:
            df = pd.read_parquet(io.BytesIO(contents))
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid Parquet file")
        _validate_columns(set(df.columns), dataset_type)
        row_count = len(df)
        csv_buf = io.BytesIO()
        df.to_csv(csv_buf, index=False)
        store_bytes = csv_buf.getvalue()

    elif ext == "json":
        import json as _json
        try:
            records = _json.loads(contents)
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid JSON file")
        if not isinstance(records, list) or not records:
            raise HTTPException(status_code=400, detail="JSON must be a non-empty array of objects")
        _validate_columns(set(records[0].keys()), dataset_type)
        row_count = len(records)
        import pandas as pd
        df = pd.DataFrame(records)
        csv_buf = io.BytesIO()
        df.to_csv(csv_buf, index=False)
        store_bytes = csv_buf.getvalue()

    else:  # csv
        import csv
        try:
            reader = csv.reader(io.StringIO(contents.decode("utf-8")))
            header = next(reader)
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid CSV file")
        _validate_columns({col.strip() for col in header}, dataset_type)
        row_count = sum(1 for _ in csv.reader(io.StringIO(contents.decode("utf-8")))) - 1
        store_bytes = contents

    target_name = "iot_readings.csv" if dataset_type == "iot" else "noaa_crw_sample.csv"
    _write_bronze(target_name, store_bytes)

    return {
        "status": "accepted",
        "dataset_type": dataset_type,
        "filename": filename,
        "format": ext,
        "target": f"bronze/{target_name}",
        "rows": row_count,
    }


class IngestEvent(BaseModel):
    reef_id: str
    timestamp: str | None = None
    water_temperature_c: float
    ph: float
    salinity_psu: float
    turbidity_ntu: float
    dissolved_oxygen_mg_l: float


class IngestBatch(BaseModel):
    events: list[IngestEvent] = Field(..., min_length=1, max_length=1000)


@app.post(
    "/ingest/stream",
    dependencies=[Depends(require_permission(Permission.UPLOAD_DATASET))],
)
def ingest_stream(batch: IngestBatch) -> dict[str, Any]:
    """Push IoT readings to the streaming pipeline (Kafka/Redpanda).

    Each event is published to the ``reef.iot.readings`` topic and validated
    against the IoT schema. Invalid events go to the dead-letter queue.

    Requires scientist or reef_admin role.
    """
    import os
    from datetime import datetime, timezone

    from infrastructure.streaming.queue import Event, get_producer
    from infrastructure.streaming.validation import run_validated_pipeline

    records = []
    for evt in batch.events:
        rec = evt.model_dump()
        if not rec.get("timestamp"):
            rec["timestamp"] = datetime.now(timezone.utc).isoformat()
        records.append(rec)

    valid_records, stats = run_validated_pipeline(records, schema="iot")

    backend = os.getenv("REEFTWIN_STREAM_BACKEND", "memory")
    producer = get_producer(backend)
    for rec in valid_records:
        producer.send(Event(
            topic="reef.iot.readings",
            key=rec["reef_id"],
            value=rec,
        ))
    producer.flush()

    return {
        "status": "accepted",
        "backend": backend,
        "total": stats.total_records,
        "valid": stats.valid_records,
        "rejected": stats.invalid_records,
        "success_rate": round(stats.success_rate, 4),
    }


@app.post(
    "/simulate",
    dependencies=[Depends(require_permission(Permission.SIMULATE))],
)
def simulate(req: SimulationRequest) -> dict[str, Any]:
    start = perf_counter()
    SIMULATION_REQUESTS.inc()
    try:
        base_state = None
        for state in get_state_store().load_states():
            if state["reef_id"] == req.reef_id:
                base_state = state
                break
        if base_state is None:
            raise HTTPException(status_code=404, detail="Reef state not found")

        base_risk = float(base_state["bleaching_risk_score"])
        temp_pressure = max(0, req.temperature_delta_c) * settings.sim_temperature_weight
        duration_pressure = min(req.duration_days / 90, 1.0) * settings.sim_duration_weight
        turbidity_pressure = max(0, req.turbidity_delta_pct) / 100 * settings.sim_turbidity_weight
        acidification_pressure = max(0, -req.ph_delta) * settings.sim_acidification_weight
        projected_risk = min(1.0, base_risk + temp_pressure + duration_pressure + turbidity_pressure + acidification_pressure)

        t = settings.risk_thresholds
        status = "stable"
        if projected_risk >= t.alert: status = "critical"
        elif projected_risk >= t.warning: status = "stressed"
        elif projected_risk >= t.watch: status = "watch"

        return {
            "reef_id": req.reef_id, "baseline_risk": base_risk,
            "projected_bleaching_risk": round(projected_risk, 4),
            "projected_ecosystem_status": status, "scenario": req.model_dump(),
        }
    finally:
        REQUEST_LATENCY.labels(endpoint="simulate").observe(perf_counter() - start)


@app.post(
    "/rag",
    dependencies=[Depends(require_permission(Permission.RAG_QUERY)), Depends(check_rate_limit)],
)
def rag_query(req: RAGQuery) -> dict[str, Any]:
    validate_query_length(req.question)
    start = perf_counter()
    RAG_QUERIES.inc()
    try:
        from infrastructure.genai.rag import HybridRAGPipeline
        pipeline = HybridRAGPipeline()
        result = pipeline.query(req.question, k=req.k)
        return {
            "answer": result.answer, "sources": result.sources,
            "model": result.model, "retrieval_method": result.retrieval_method,
            "tokens": {"input": result.input_tokens, "output": result.output_tokens},
        }
    finally:
        REQUEST_LATENCY.labels(endpoint="rag").observe(perf_counter() - start)


@app.post(
    "/agent",
    dependencies=[Depends(require_permission(Permission.AGENT_QUERY)), Depends(check_rate_limit)],
)
def agent_query(req: AgentQuery) -> dict[str, Any]:
    validate_query_length(req.query)
    start = perf_counter()
    AGENT_QUERIES.inc()
    try:
        from infrastructure.genai.agent import ReefAgent
        agent = ReefAgent()
        result = agent.run(req.query)
        for tc in result.tool_calls:
            AGENT_TOOL_CALLS.labels(tool_name=tc.get("tool", "unknown")).inc()
        return {
            "answer": result.answer, "tool_calls": result.tool_calls,
            "iterations": result.iterations,
            "tokens": {"input": result.total_input_tokens, "output": result.total_output_tokens},
        }
    finally:
        REQUEST_LATENCY.labels(endpoint="agent").observe(perf_counter() - start)


@app.post(
    "/interpret",
    dependencies=[Depends(require_permission(Permission.INTERPRET))],
)
def interpret(req: SimulationRequest) -> dict[str, Any]:
    start = perf_counter()
    try:
        sim_result = simulate(req)
        from infrastructure.genai.scenario_interpreter import interpret_simulation
        interpretation = interpret_simulation(sim_result)
        return {
            "simulation": sim_result,
            "interpretation": {
                "summary": interpretation.summary,
                "risk_assessment": interpretation.risk_assessment,
                "recommendations": interpretation.recommendations,
            },
            "model": interpretation.model,
        }
    finally:
        REQUEST_LATENCY.labels(endpoint="interpret").observe(perf_counter() - start)


@app.post(
    "/query",
    dependencies=[Depends(require_permission(Permission.SMART_QUERY))],
)
def smart_query(req: QueryRequest) -> dict[str, Any]:
    start = perf_counter()
    try:
        from infrastructure.genai.router import route_and_execute
        return route_and_execute(req.query)
    finally:
        REQUEST_LATENCY.labels(endpoint="query").observe(perf_counter() - start)
