# Changelog

## 0.1.0 (2026-05-07)

Initial release with all 7 tiers complete.

### Features
- Digital twin state engine with pluggable backends (JSON, S3)
- Bleaching risk prediction (RandomForest + Physics-Informed ML hybrid + PINN)
- Uncertainty quantification via conformal prediction
- Multi-objective stress scoring (4-dimensional, weighted)
- Anomaly detection (Isolation Forest)
- Time-series forecasting (Holt-Winters, SARIMA, Prophet — pluggable)
- Coral vision classifier (feature-based + CNN)
- Edge deployment (ONNX export + lightweight numpy predictor)
- Fallback chains with heuristic predictor
- Reef knowledge RAG (hybrid search: BM25 + dense + RRF)
- ReAct decision-support agent with 4 reef tools
- LLM scenario interpretation (natural-language summaries)
- Query complexity routing (simple/moderate/complex)
- Pluggable LLM providers (Claude, OpenAI/Codex, Qwen, Ollama, Mock)
- Pluggable vector stores (memory, Qdrant, pgvector, Milvus)
- S3-compatible storage (SeaweedFS, MinIO, AWS S3)
- MLflow experiment tracking + inference tracing
- Evidently AI drift monitoring + custom PSI monitor
- Benchmark comparison tool (before/after with percentiles)
- AI governance (model cards, audit trails, data lineage)
- Streamlit dashboard (4 pages)
- React frontend (5 pages, Vite + TypeScript + Tailwind)
- Grafana dashboard provisioning (8 panels)
- 8 Prometheus metrics
- Streaming event queue (in-memory + Kafka/Redpanda)
- Schema validation + Dead-Letter Queue + retries
- KFP v2 pipeline components
- SageMaker integration wrappers
- GitHub Actions CI/CD
- API security (key auth, rate limiting, input validation)
- Real NOAA CRW API integration (with synthetic fallback)
- Ecosystem graph (NetworkX, 13 nodes, stress propagation)
- FTI architecture (Feature/Training/Inference as independent pipelines)

### Experiments
- Pipeline latency: -92.9% (29ms → 2ms)
- Inference cost: -95.5% (200 → 9 invocations)
- Pipeline reliability: 6 bad records caught and quarantined
