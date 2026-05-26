# ReefTwin — Quick Start

## Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) (recommended) or pip
- Docker & Docker Compose (for full stack)

## 1. Auth Modes

Controlled by `REEFTWIN_AUTH_MODE` in `.env` or environment:

| Mode | How to use | Notes |
|------|-----------|-------|
| `none` | `export REEFTWIN_AUTH_MODE=none` | Dev mode, all endpoints open, user gets `reef_admin` role |
| `apikey` (default) | `export REEFTWIN_API_KEYS=my-secret-key` | Pass `X-API-Key: my-secret-key` header; gets `reef_admin` role |
| `oidc` | Set `OIDC_ISSUER_URL`, `OIDC_AUDIENCE` | JWT Bearer token; roles from `reeftwin_roles` claim |

**Quick dev test (no auth):**

```bash
REEFTWIN_AUTH_MODE=none make run-api
curl http://localhost:8000/reefs
curl http://localhost:8000/public/reefs
```

**API key test:**

```bash
REEFTWIN_AUTH_MODE=apikey REEFTWIN_API_KEYS=test123 make run-api
curl -H "X-API-Key: test123" http://localhost:8000/reefs
curl -X POST -H "X-API-Key: test123" -H "Content-Type: application/json" \
  -d '{"reef_id":"gbr_heron_reef"}' http://localhost:8000/simulate
```

## 2. RBAC (OIDC mode)

When using OIDC, include `reeftwin_roles` in the JWT claims from your IdP (Keycloak, Auth0, Entra ID):

```json
{
  "sub": "user-42",
  "reeftwin_roles": ["scientist"],
  "iss": "https://keycloak.example.com/realms/reeftwin",
  "aud": "reeftwin-api"
}
```

Keycloak's `realm_access.roles` convention is also supported.

**Roles:**

| Role | Access |
|------|--------|
| `reef_admin` | Full access to all resources |
| `scientist` | Simulate, upload datasets, view all reef states, RAG/agent queries |
| `analyst` | View reef states and dashboards, RAG queries (read-only, no mutations) |
| `public_viewer` | Only `/public/reefs` (summary data, no auth required for that endpoint) |

Deny-by-default: tokens with no recognised role get `403 Forbidden`.

## 3. Local Development (API only)

```bash
cd ReefTwin

# Create venv and install
uv venv .venv && source .venv/bin/activate
uv pip install -e ".[dev,genai]"

# Copy env config (defaults work for local dev)
cp .env.example .env

# Run the FTI pipeline to generate data
make generate-sample-data    # -> data/bronze/iot_readings.csv
make ingest-noaa             # -> data/bronze/noaa_crw_sample.csv
make build-features          # -> data/gold/reef_features.csv
make train-model             # -> models/bleaching_risk/model.joblib
make update-twin             # -> data/gold/reef_state.json

# Start the API (hot reload)
make run-api
```

API available at:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

Optional local extras:

```bash
uv pip install -e ".[mlops]"        # MLflow tracking + Evidently drift
uv pip install -e ".[dashboard]"    # Streamlit dashboard
uv pip install -e ".[torch]"        # PINN / CNN models
uv pip install -e ".[s3]"           # S3-compatible state store

make run-dashboard     # -> http://localhost:8501
make run-frontend      # -> http://localhost:3001
make run-experiments   # MLflow experiments
```

## 4. Full Stack (Docker Compose)

The `Dockerfile` accepts an `INSTALL_EXTRAS` build arg to control which optional dependency groups are installed. Defaults to core-only (`"."`).

```bash
# Dev stack (core deps): API + Redpanda + Prometheus + Grafana + Console
docker compose up --build

# Dev stack with GenAI + MLOps + S3 extras:
INSTALL_EXTRAS=".[genai,mlops,s3]" docker compose up --build

# Prod stack: SASL-enabled Redpanda, no Console UI
docker compose -f docker-compose.prod.yml up --build
```

To pass the build arg, add it to `docker-compose.yml` under the `reeftwin-api` service:

```yaml
services:
  reeftwin-api:
    build:
      context: .
      args:
        INSTALL_EXTRAS: ".[genai,mlops,s3]"
```

Or build the image directly:

```bash
docker build --build-arg INSTALL_EXTRAS=".[genai,mlops,s3]" -t reeftwin-api:latest .
```

Available extras: `genai`, `mlops`, `s3`, `dashboard`, `torch`, `openai`, `qdrant`, `pgvector`, `milvus`, `forecasting`, `kfp`, `sagemaker`, `dev`.

| Service | URL |
|---------|-----|
| ReefTwin API | http://localhost:8000 |
| Swagger Docs | http://localhost:8000/docs |
| Redpanda Console | http://localhost:8080 (dev only) |
| Prometheus | http://localhost:9090 |
| Grafana | http://localhost:3000 |

### Running the FTI pipeline in Docker

The image does not run the FTI pipeline at build time. If you built without running the pipeline locally first, the container starts with no data (`/ready` returns `not_ready`).

**Option A — Run locally before building** (data baked into image via `COPY . .`):

```bash
make generate-sample-data && make ingest-noaa && make build-features && make train-model && make update-twin
docker compose up --build
```

**Option B — Run inside the running container:**

```bash
docker compose up -d --build
docker compose exec reeftwin-api bash -c "\
  python -m pipelines.simulate_iot_stream && \
  python -m pipelines.ingest_noaa_crw && \
  python -m pipelines.build_features && \
  python -m models.bleaching_risk.train && \
  python -m pipelines.update_twin_state"
```

**Option C — Use real NOAA data** (fetches from CoastWatch ERDDAP API):

```bash
docker compose exec reeftwin-api python -m pipelines.ingest_noaa_real
```

## 5. Deploying on Kubeflow4X

ReefTwin integrates with the [Kubeflow4X](https://github.com/AduraX/Kubeflow4X) platform, reusing its Keycloak OIDC, SeaweedFS S3, KFP pipelines, and KServe model serving. See the [KF4X articles](docs/articles/) for full platform documentation.

### What maps where

| ReefTwin need | KF4X provides |
|---|---|
| OIDC auth + RBAC | Keycloak realm `kubeflow-4x` at `keycloak.util.lcl` |
| S3 state store | SeaweedFS (per-namespace buckets via PodDefault) |
| MLflow tracking | Phase 2 MLflow server at `mlflow.k8s.lcl` |
| Model serving | KServe (Phase 1, with `sa-s3-kserve` ServiceAccount) |
| Pipelines (FTI) | Kubeflow Pipelines (KFP v2) |
| Dashboards | Phase 3 Superset + DuckDB |

### 5a. Build the image

Build with the extras needed for Kubeflow (uses the `INSTALL_EXTRAS` build arg, see also section 4):

```bash
docker build --build-arg INSTALL_EXTRAS=".[genai,mlops,s3,kfp]" -t kf4x/reeftwin-api:latest .

# For Kind clusters:
kind load docker-image kf4x/reeftwin-api:latest --name <cluster-name>
```

### 5b. Add a ReefTwin user profile

Append to `Kubeflow4x_Phase-1/profiles/users.csv`:

```csv
reeftwin@util.lcl,reeftwin@util.lcl,owner,kf4x-user-grp
```

Re-run `manage-users.sh` to create the `reeftwin` namespace, Profile, ResourceQuota, and S3 credentials.

### 5c. Register RBAC roles in Keycloak

In the `kubeflow-4x` realm, create four realm roles:

```
reef_admin
scientist
analyst
public_viewer
```

Add a **client scope mapper** on the `kubeflow-client` OIDC client:

- Mapper type: **User Realm Role**
- Token claim name: `reeftwin_roles`
- Claim JSON type: `String`
- Multivalued: `true`
- Add to ID token and access token: `true`

Assign roles to existing users:

| User | Role |
|------|------|
| `admin@util.lcl` | `reef_admin` |
| `dev.edit@util.lcl` | `scientist` |
| `analyst@util.lcl` | `analyst` |
| `viewer@util.lcl` | `public_viewer` |
| `auditor@util.lcl` | `analyst` |

### 5d. Deploy to Kubernetes

```yaml
# k8s/reeftwin-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: reeftwin-api
  namespace: reeftwin
spec:
  replicas: 1
  selector:
    matchLabels:
      app: reeftwin-api
  template:
    metadata:
      labels:
        app: reeftwin-api
        access-ml-pipeline: "true"       # inject KFP token
        access-seaweedfs: "true"         # inject S3 creds
    spec:
      serviceAccountName: sa-s3-kserve
      containers:
      - name: api
        image: kf4x/reeftwin-api:latest
        ports:
        - containerPort: 8000
        env:
        - name: REEFTWIN_AUTH_MODE
          value: "oidc"
        - name: OIDC_ISSUER_URL
          value: "https://keycloak.util.lcl/realms/kubeflow-4x"
        - name: OIDC_AUDIENCE
          value: "kubeflow-client"
        - name: REEFTWIN_STATE_STORE_BACKEND
          value: "s3"
        - name: REEFTWIN_S3_ENDPOINT
          value: "minio-service.kubeflow:9000"
        - name: REEFTWIN_S3_BUCKET
          value: "tenant-reeftwin"
        - name: REEFTWIN_S3_USE_SSL
          value: "false"
        # S3 creds injected by access-seaweedfs PodDefault
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8000
          initialDelaySeconds: 5
        resources:
          requests:
            cpu: 500m
            memory: 512Mi
          limits:
            cpu: "2"
            memory: 2Gi
---
apiVersion: v1
kind: Service
metadata:
  name: reeftwin-api
  namespace: reeftwin
spec:
  selector:
    app: reeftwin-api
  ports:
  - port: 8000
    targetPort: 8000
```

### 5e. Expose via ingress

For Traefik (default KF4X `INGRESS_TYPE`):

```yaml
# k8s/reeftwin-httproute.yaml
apiVersion: gateway.networking.k8s.io/v1
kind: HTTPRoute
metadata:
  name: reeftwin
  namespace: reeftwin
spec:
  parentRefs:
  - name: kubeflow-gateway
    namespace: kubeflow
  hostnames:
  - "reeftwin.k8s.lcl"
  rules:
  - matches:
    - path:
        type: PathPrefix
        value: /
    backendRefs:
    - name: reeftwin-api
      port: 8000
```

Add `reeftwin.k8s.lcl` to your DNS or `/etc/hosts`.

### 5f. Apply

```bash
kubectl apply -f k8s/reeftwin-deployment.yaml
kubectl apply -f k8s/reeftwin-httproute.yaml
```

### 5g. Auth flow end-to-end

```
Browser -> keycloak.util.lcl (login)
        -> JWT with reeftwin_roles: ["scientist"]
        -> reeftwin.k8s.lcl/simulate (Bearer token)
        -> Istio validates JWT (RequestAuthentication)
        -> ReefTwin RBAC checks Permission.SIMULATE
        -> 200 OK
```

Defense-in-depth: Istio mesh validates the JWT at the network level, then ReefTwin's application RBAC enforces fine-grained permissions.

### 5h. Running the FTI pipeline on Kubeflow

**Option A — KFP pipeline** (recommended for production):

```bash
make compile-kfp    # -> pipelines/kfp/reeftwin_pipeline.yaml
```

Upload the compiled YAML via the KFP UI or submit from a notebook:

```python
import kfp
client = kfp.Client()
client.create_run_from_pipeline_package(
    "pipelines/kfp/reeftwin_pipeline.yaml",
    arguments={"iot_rows": 5000, "noaa_days": 60},
)
```

Pipeline graph: `generate_iot_data` + `generate_noaa_data` -> `build_features` -> `train_model`. Each step runs as a pod with S3 and KFP tokens injected by PodDefaults.

**Option B — Kubeflow notebook** (interactive exploration):

From a Jupyter notebook in the `reeftwin` namespace (with `access-seaweedfs` PodDefault selected):

```python
from pipelines.simulate_iot_stream import generate_readings
from pipelines.ingest_noaa_real import fetch_noaa_crw
from pipelines.build_features import build_features
from models.bleaching_risk.train import train_model
from pipelines.update_twin_state import update_state

# Generate or fetch data
iot = generate_readings(5000)
noaa = fetch_noaa_crw(days=60)  # real NOAA data with synthetic fallback

# Build features and train
features = build_features(iot_path, noaa_path)
train_model(features_path, model_path)
update_state(features_path, model_path)
```

**Option C — Kubernetes Job** (one-off initialization):

```bash
kubectl -n reeftwin run fti-init --rm -it \
  --image=kf4x/reeftwin-api:latest \
  --restart=Never -- bash -c "\
    python -m pipelines.ingest_noaa_real && \
    python -m pipelines.simulate_iot_stream && \
    python -m pipelines.build_features && \
    python -m models.bleaching_risk.train && \
    python -m pipelines.update_twin_state"
```

## 6. Data Ingestion — Real-World Datasets

### Synthetic data (for development)

```bash
make generate-sample-data    # 5000 simulated IoT readings with heat-stress scenario
make ingest-noaa             # synthetic NOAA CRW satellite data
```

### Real NOAA satellite data

Fetches SST, SST Anomaly, HotSpot, DHW, and Bleaching Alert Area from the [CoastWatch ERDDAP API](https://coastwatch.pfeg.noaa.gov/erddap/) (no API key required):

```bash
make ingest-noaa-real                          # default: 60 days, 3 reefs
python -m pipelines.ingest_noaa_real --days 90 # custom lookback
```

Falls back to synthetic data automatically if the API is unreachable.

### Uploading custom datasets

**Via API** (requires `scientist` or `reef_admin` role):

```bash
curl -X POST "http://localhost:8000/datasets/upload?reef_id=gbr_heron_reef" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"records": [{"water_temperature_c": 29.1, "ph": 8.05, "salinity_psu": 35.2}]}'
```

**Via S3** (direct upload to bronze layer):

Upload CSV files matching the expected schema to the bronze S3 prefix:

```bash
# Local (json state store)
cp my_iot_data.csv data/bronze/iot_readings.csv
cp my_noaa_data.csv data/bronze/noaa_crw_sample.csv

# S3 (SeaweedFS / MinIO / AWS)
aws s3 cp my_iot_data.csv s3://tenant-reeftwin/bronze/iot_readings.csv \
  --endpoint-url http://minio-service.kubeflow:9000
```

**Expected CSV schemas:**

IoT readings (`data/bronze/iot_readings.csv`):

| Column | Type | Example |
|--------|------|---------|
| `reef_id` | string | `gbr_heron_reef` |
| `timestamp` | ISO datetime | `2026-05-07T10:30:00` |
| `water_temperature_c` | float | `28.3` |
| `ph` | float | `8.05` |
| `salinity_psu` | float | `35.1` |
| `turbidity_ntu` | float | `0.8` |
| `dissolved_oxygen_mg_l` | float | `6.5` |

NOAA CRW data (`data/bronze/noaa_crw_sample.csv`):

| Column | Type | Example |
|--------|------|---------|
| `reef_id` | string | `gbr_heron_reef` |
| `date` | date | `2026-05-07` |
| `sst_celsius` | float | `28.9` |
| `sst_anomaly_c` | float | `0.5` |
| `hotspot_c` | float | `0.3` |
| `degree_heating_weeks` | float | `2.1` |
| `bleaching_alert_area` | int | `1` |

After uploading, re-run the pipeline from `build-features` onward:

```bash
make build-features && make train-model && make update-twin
```

## 7. Tests


```bash
# All tests
make test

# Specific test suites
pytest tests/test_rbac.py -v            # RBAC authorization (54 tests)
pytest tests/test_oidc_auth.py -v       # OIDC/JWT auth (22 tests)
pytest tests/test_security_risks.py -v  # Security risk treatments
pytest tests/test_integration.py -v     # Full pipeline + API contracts
```

## 8. API Endpoints

```
Public (no auth):
  GET  /health                    Health check
  GET  /public/reefs              Public reef summaries
  GET  /metrics                   Prometheus metrics

Protected (analyst+):
  GET  /reefs                     All reef states
  GET  /reefs/{reef_id}/state     Single reef state (object-level ACL)
  POST /rag                       RAG query
  POST /interpret                 Interpret simulation results
  POST /query                     Smart query router

Protected (scientist+):
  POST /simulate                  Run simulation
  POST /datasets/upload?reef_id=  Upload dataset
  POST /agent                     Agent query
```
