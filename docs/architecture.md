# ReefTwin Architecture

ReefTwin is designed as a production-style digital twin platform for coral reefs.

## Layers

1. Data sources: NOAA Coral Reef Watch, AIMS, Allen Coral Atlas, simulated IoT sensors.
2. Ingestion: streaming and batch ingestion with schema validation.
3. Processing: feature engineering for reef heat stress, water quality, and trend metrics.
4. Twin state: current live reef state persisted as queryable JSON/database records.
5. AI: bleaching risk prediction, anomaly detection, forecasting, coral vision.
6. Serving: FastAPI endpoints, dashboard, Prometheus metrics.
7. Operations: tests, health checks, Kubernetes starter manifests.

## Digital twin definition

A reef digital twin is not just a dashboard. It must maintain a live state, update that state from observations, and support scenario simulation.

