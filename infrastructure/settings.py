from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import Field
from pydantic_settings import BaseSettings


_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with open(path) as f:
        return yaml.safe_load(f) or {}


def _load_reef_ids(path: Path) -> list[str]:
    data = _load_yaml(path)
    return [r["reef_id"] for r in data.get("reefs", [])]


def _load_model_config(path: Path) -> dict[str, Any]:
    data = _load_yaml(path)
    return data.get("bleaching_risk", {})


class RiskThresholds(BaseSettings):
    alert: float = 0.85
    warning: float = 0.70
    watch: float = 0.50
    normal: float = 0.0


class ReefTwinSettings(BaseSettings):
    # --- Paths ---
    state_path: Path = Field(
        default=_PROJECT_ROOT / "data" / "gold" / "reef_state.json",
        alias="REEFTWIN_STATE_PATH",
    )
    model_path: Path = Field(
        default=_PROJECT_ROOT / "models" / "bleaching_risk" / "model.joblib",
        alias="REEFTWIN_MODEL_PATH",
    )
    features_path: Path = Field(
        default=_PROJECT_ROOT / "data" / "gold" / "reef_features.parquet",
        alias="REEFTWIN_FEATURES_PATH",
    )
    iot_output: Path = Field(
        default=_PROJECT_ROOT / "data" / "bronze" / "iot_readings.csv",
        alias="REEFTWIN_IOT_OUTPUT",
    )
    noaa_output: Path = Field(
        default=_PROJECT_ROOT / "data" / "bronze" / "noaa_crw_sample.csv",
        alias="REEFTWIN_NOAA_OUTPUT",
    )

    # --- State Store Backend ---
    state_store_backend: str = Field(default="json", alias="REEFTWIN_STATE_STORE_BACKEND")

    # --- Reef IDs (loaded from configs/reefs.yml) ---
    reef_ids: list[str] = Field(default_factory=lambda: _load_reef_ids(_PROJECT_ROOT / "configs" / "reefs.yml"))

    # --- Model Config (loaded from configs/model_config.yml) ---
    features: list[str] = Field(
        default_factory=lambda: _load_model_config(_PROJECT_ROOT / "configs" / "model_config.yml").get(
            "features",
            [
                "water_temperature_c", "ph", "salinity_psu", "turbidity_ntu",
                "dissolved_oxygen_mg_l", "sst_anomaly_c", "hotspot_c",
                "degree_heating_weeks", "temperature_trend_7d",
            ],
        )
    )
    target: str = "bleaching_label"

    # --- Risk Thresholds (single source of truth) ---
    risk_thresholds: RiskThresholds = Field(default_factory=RiskThresholds)

    # --- Bleaching Label Thresholds ---
    bleaching_dhw_threshold: float = 4.0
    bleaching_temp_threshold: float = 30.0
    bleaching_hotspot_threshold: float = 1.0

    # --- Simulation Weights ---
    sim_temperature_weight: float = 0.12
    sim_duration_weight: float = 0.08
    sim_turbidity_weight: float = 0.08
    sim_acidification_weight: float = 0.25

    # --- S3-Compatible Storage (SeaweedFS / MinIO / AWS S3) ---
    s3_endpoint: str = Field(default="", alias="REEFTWIN_S3_ENDPOINT")
    s3_bucket: str = Field(default="reeftwin", alias="REEFTWIN_S3_BUCKET")
    s3_access_key: str = Field(default="", alias="REEFTWIN_S3_ACCESS_KEY")
    s3_secret_key: str = Field(default="", alias="REEFTWIN_S3_SECRET_KEY")
    s3_region: str = Field(default="us-east-1", alias="REEFTWIN_S3_REGION")
    s3_use_ssl: bool = Field(default=False, alias="REEFTWIN_S3_USE_SSL")

    # --- Vector Store Backend ---
    vector_store_backend: str = Field(
        default="memory",
        alias="REEFTWIN_VECTOR_STORE_BACKEND",
        description="Vector store backend: memory, qdrant, pgvector, milvus",
    )
    # Qdrant
    qdrant_host: str = Field(default="localhost", alias="QDRANT_HOST")
    qdrant_port: int = Field(default=6333, alias="QDRANT_PORT")
    qdrant_api_key: str = Field(default="", alias="QDRANT_API_KEY")
    qdrant_collection: str = Field(default="reef_knowledge", alias="QDRANT_COLLECTION")
    qdrant_prefer_grpc: bool = Field(default=False, alias="QDRANT_PREFER_GRPC")
    # pgvector
    pgvector_dsn: str = Field(default="postgresql://localhost:5432/reeftwin", alias="PGVECTOR_DSN")
    pgvector_table: str = Field(default="reef_embeddings", alias="PGVECTOR_TABLE")
    pgvector_index_type: str = Field(default="hnsw", alias="PGVECTOR_INDEX_TYPE")
    pgvector_hnsw_m: int = Field(default=16, alias="PGVECTOR_HNSW_M")
    pgvector_hnsw_ef: int = Field(default=64, alias="PGVECTOR_HNSW_EF")
    pgvector_probes: int = Field(default=10, alias="PGVECTOR_PROBES")
    # Milvus
    milvus_host: str = Field(default="localhost", alias="MILVUS_HOST")
    milvus_port: int = Field(default=19530, alias="MILVUS_PORT")
    milvus_collection: str = Field(default="reef_knowledge", alias="MILVUS_COLLECTION")
    milvus_index_type: str = Field(default="IVF_FLAT", alias="MILVUS_INDEX_TYPE")
    milvus_nlist: int = Field(default=128, alias="MILVUS_NLIST")
    milvus_nprobe: int = Field(default=16, alias="MILVUS_NPROBE")
    milvus_metric_type: str = Field(default="COSINE", alias="MILVUS_METRIC_TYPE")

    # --- LLM Provider ---
    llm_provider: str = Field(
        default="claude",
        alias="REEFTWIN_LLM_PROVIDER",
        description="LLM provider: claude, openai, qwen, ollama, mock",
    )
    llm_model: str = Field(
        default="claude-sonnet-4-20250514",
        alias="REEFTWIN_LLM_MODEL",
        description="Model ID to use (provider-specific)",
    )
    llm_temperature: float = Field(default=0.3, alias="REEFTWIN_LLM_TEMPERATURE")
    llm_max_tokens: int = Field(default=1024, alias="REEFTWIN_LLM_MAX_TOKENS")
    # OpenAI / Codex
    openai_api_key: str = Field(default="", alias="OPENAI_API_KEY")
    openai_base_url: str = Field(default="https://api.openai.com/v1", alias="OPENAI_BASE_URL")
    # Qwen (via OpenAI-compatible endpoint — DashScope or local)
    qwen_api_key: str = Field(default="", alias="QWEN_API_KEY")
    qwen_base_url: str = Field(
        default="https://dashscope.aliyuncs.com/compatible-mode/v1",
        alias="QWEN_BASE_URL",
    )
    # Ollama (local)
    ollama_base_url: str = Field(default="http://localhost:11434", alias="OLLAMA_BASE_URL")

    # --- Logging ---
    log_level: str = Field(default="INFO", alias="REEFTWIN_LOG_LEVEL")

    model_config = {"env_prefix": "REEFTWIN_", "populate_by_name": True}


settings = ReefTwinSettings()


def read_df(path: str | Path) -> "pd.DataFrame":
    """Read a CSV or Parquet file into a pandas DataFrame based on extension."""
    import pandas as pd
    path = str(path)
    if path.endswith(".parquet"):
        return pd.read_parquet(path)
    return pd.read_csv(path)
