# ReefTwin Threat Model

> **Methodology:** STRIDE (Spoofing, Tampering, Repudiation, Information Disclosure, Denial of Service, Elevation of Privilege)
> **Scope:** All system components as deployed in production (Docker Compose / Kubernetes / KF4X)
> **Status:** Security-aligned with OWASP API Security Top 10 (2023), CIS Kubernetes Benchmark, and NIST AI RMF 1.0
> **Version:** 0.1.0
> **Last reviewed:** 2026-05-07

---

## System Boundary

```
                    ┌─ Internet ─────────────────────────────┐
                    │                                         │
                    │  Users (scientists, reef managers)       │
                    │  React frontend / Streamlit dashboard    │
                    └──────────────┬──────────────────────────┘
                                   │ HTTPS
                    ┌──────────────▼──────────────────────────┐
                    │  Ingress (Traefik / Nginx / ALB)         │
                    │  TLS termination                         │
                    └──────────────┬──────────────────────────┘
                                   │
        ┌──────────────────────────▼───────────────────────────┐
        │                    Trust Boundary                      │
        │  ┌─────────────┐  ┌──────────────┐  ┌─────────────┐  │
        │  │ FastAPI      │  │ Redpanda     │  │ Prometheus  │  │
        │  │ (9 endpoints)│  │ (Kafka)      │  │ + Grafana   │  │
        │  └──────┬───────┘  └──────┬───────┘  └─────────────┘  │
        │         │                 │                             │
        │  ┌──────▼───────┐  ┌─────▼────────┐                   │
        │  │ ML Models    │  │ Streaming    │                    │
        │  │ (RF, PIML,   │  │ Consumer     │                    │
        │  │  PINN, CNN)  │  │ + Validator  │                    │
        │  └──────┬───────┘  └──────────────┘                    │
        │         │                                              │
        │  ┌──────▼───────┐  ┌──────────────┐                   │
        │  │ State Store  │  │ S3 Storage   │                    │
        │  │ (JSON / S3)  │  │ (SeaweedFS)  │                    │
        │  └──────────────┘  └──────────────┘                    │
        │                                                        │
        │  ┌──────────────┐  ┌──────────────┐                   │
        │  │ LLM Provider │  │ MLflow       │                    │
        │  │ (Claude/Qwen)│  │ Tracking     │                    │
        │  └──────────────┘  └──────────────┘                    │
        └────────────────────────────────────────────────────────┘
```

---

## 1. Reef State API (`/reefs`, `/reefs/{reef_id}/state`)

### 1.1 Spoofing

| Threat | Description | Likelihood | Impact | Mitigation |
|--------|-------------|-----------|--------|------------|
| T-API-S1 | Unauthenticated user queries reef state data | Medium | Low | API key auth available. Read endpoints are low-sensitivity (reef state is environmental data, not PII) |
| T-API-S2 | Attacker spoofs API key to access LLM endpoints | Low | Medium | API keys loaded from env var. Rate limiting (30/min) bounds abuse. Rotate keys on suspected compromise |

### 1.2 Tampering

| Threat | Description | Likelihood | Impact | Mitigation |
|--------|-------------|-----------|--------|------------|
| T-API-T1 | Attacker modifies reef state JSON on disk/S3 | Low | High | File-based store relies on OS/S3 permissions. S3 backend uses IAM-scoped credentials. Audit trail tracks state updates |
| T-API-T2 | Man-in-the-middle modifies API responses | Low | Medium | TLS at ingress. Internal cluster traffic via Istio mTLS on KF4X |

### 1.3 Repudiation

| Threat | Description | Likelihood | Impact | Mitigation |
|--------|-------------|-----------|--------|------------|
| T-API-R1 | User denies making a simulation request | Low | Low | Structured logging captures all requests with timestamps. Prediction audit trail records every inference |

### 1.4 Information Disclosure

| Threat | Description | Likelihood | Impact | Mitigation |
|--------|-------------|-----------|--------|------------|
| T-API-I1 | Reef state data exposed to unauthorised users | Medium | Low | Reef data is environmental (not PII). API key required on sensitive endpoints. CORS restricts browser access to localhost:3001 |
| T-API-I2 | Error responses leak internal details | Low | Low | FastAPI production mode suppresses stack traces. Custom HTTPException messages |

### 1.5 Denial of Service

| Threat | Description | Likelihood | Impact | Mitigation |
|--------|-------------|-----------|--------|------------|
| T-API-D1 | Flood of requests overwhelms API | Medium | Medium | Rate limiter (30/min) on LLM endpoints. Kubernetes HPA scales pods. Resource limits (1 CPU, 1Gi) prevent single-pod resource exhaustion |
| T-API-D2 | Large query payload consumes memory | Low | Medium | Query length validated (max 2000 chars). Pydantic enforces `duration_days` range (1-365) |

### 1.6 Elevation of Privilege

| Threat | Description | Likelihood | Impact | Mitigation |
|--------|-------------|-----------|--------|------------|
| T-API-E1 | API key grants access to all endpoints | Medium | Low | Single-tier auth. No privilege escalation possible (no admin actions via API). Role-based access planned |

---

## 2. Simulation API (`/simulate`, `/interpret`)

### 2.1 Tampering

| Threat | Description | Likelihood | Impact | Mitigation |
|--------|-------------|-----------|--------|------------|
| T-SIM-T1 | Attacker crafts adversarial simulation parameters to produce misleading risk scores | Medium | High | Pydantic validates parameter ranges. Stress weights from Pydantic Settings (not user-controllable). Simulation uses server-side physics model, not user-supplied formulas |
| T-SIM-T2 | Prompt injection in `/interpret` (LLM interprets simulation result) | Medium | Medium | LLM receives structured JSON, not raw user input. System prompt constrains output format. No tool execution from interpretation |

### 2.2 Denial of Service

| Threat | Description | Likelihood | Impact | Mitigation |
|--------|-------------|-----------|--------|------------|
| T-SIM-D1 | Expensive simulation parameters (365-day duration) consume compute | Low | Medium | `duration_days` capped at 365 via Pydantic. Simulation is O(1) math, not iterative — cost is constant regardless of duration |

### 2.3 Information Disclosure

| Threat | Description | Likelihood | Impact | Mitigation |
|--------|-------------|-----------|--------|------------|
| T-SIM-I1 | Simulation results reveal internal model weights | Low | Low | API returns projected risk score, not model parameters. Weights are in settings, not in response |

---

## 3. Ingestion Pipelines

### 3.1 Tampering

| Threat | Description | Likelihood | Impact | Mitigation |
|--------|-------------|-----------|--------|------------|
| T-ING-T1 | Poisoned NOAA data injected via man-in-the-middle on CoastWatch API | Low | High | NOAA URL hardcoded (not user-supplied). HTTPS used for ERDDAP API. Schema validation (Pydantic `NOAARecordSchema`) rejects out-of-range values. Falls back to synthetic data on failure |
| T-ING-T2 | Malformed IoT sensor readings corrupt feature pipeline | Medium | Medium | `IoTReadingSchema` validates all fields (type, range). Invalid records quarantined in Dead-Letter Queue. Pipeline reports success rate |

### 3.2 Repudiation

| Threat | Description | Likelihood | Impact | Mitigation |
|--------|-------------|-----------|--------|------------|
| T-ING-R1 | Source of bad data cannot be traced | Low | Medium | DLQ entries include: timestamp, source, error, full record, attempt count. Structured logs trace pipeline execution |

### 3.3 Denial of Service

| Threat | Description | Likelihood | Impact | Mitigation |
|--------|-------------|-----------|--------|------------|
| T-ING-D1 | NOAA API unavailable prevents data refresh | Medium | Medium | Automatic fallback to synthetic data generation. No hard dependency on external API. Stale data triggers drift alert (Evidently/PSI) |
| T-ING-D2 | IoT event flood overwhelms streaming consumer | Low | Medium | In-memory queue is bounded by Python process memory. Kafka/Redpanda handles backpressure via consumer groups. Schema validation rejects invalid events early |

---

## 4. Kafka/Redpanda Topics

### 4.1 Spoofing

| Threat | Description | Likelihood | Impact | Mitigation |
|--------|-------------|-----------|--------|------------|
| T-KAF-S1 | Unauthorised producer publishes to `reef.iot.readings` topic | Medium | High | Redpanda internal port (9092) not exposed externally. External access (19092) is development-only. Production should enable Redpanda ACLs or SASL authentication |

### 4.2 Tampering

| Threat | Description | Likelihood | Impact | Mitigation |
|--------|-------------|-----------|--------|------------|
| T-KAF-T1 | Messages tampered in transit | Low | High | Internal cluster traffic. Redpanda supports TLS for inter-broker and client communication (not enabled by default in dev compose). Production should enable TLS |

### 4.3 Information Disclosure

| Threat | Description | Likelihood | Impact | Mitigation |
|--------|-------------|-----------|--------|------------|
| T-KAF-I1 | Redpanda Console exposes topic data to unauthorised users | Medium | Medium | Console exposed on port 8080 in dev compose. Production should require authentication or disable Console |

### 4.4 Denial of Service

| Threat | Description | Likelihood | Impact | Mitigation |
|--------|-------------|-----------|--------|------------|
| T-KAF-D1 | Topic filled with invalid messages | Medium | Medium | Schema validation at consumer rejects invalid messages. DLQ captures failures. Consumer group rebalancing handles slow consumers |

---

## 5. Object Storage (S3 / SeaweedFS)

### 5.1 Spoofing

| Threat | Description | Likelihood | Impact | Mitigation |
|--------|-------------|-----------|--------|------------|
| T-S3-S1 | Attacker uses stolen S3 credentials to access reef data | Medium | High | Credentials in env vars (not source control). KF4X provides per-namespace IAM with scoped policies. AWS S3 supports IAM roles (no long-lived keys) |

### 5.2 Tampering

| Threat | Description | Likelihood | Impact | Mitigation |
|--------|-------------|-----------|--------|------------|
| T-S3-T1 | Attacker modifies model artefacts in S3 | Low | Critical | MLflow Model Registry tracks model versions. Model cards include training metadata. Production should enable S3 versioning and MFA delete |
| T-S3-T2 | Attacker modifies gold-layer feature data | Low | High | Feature pipeline is idempotent (re-runnable from bronze). Data lineage graph traces provenance. Iceberg tables (KF4X Phase 3) provide time-travel for recovery |

### 5.3 Information Disclosure

| Threat | Description | Likelihood | Impact | Mitigation |
|--------|-------------|-----------|--------|------------|
| T-S3-I1 | S3 bucket publicly accessible | Low | Medium | SeaweedFS default is private. AWS S3 should use `BlockPublicAccess`. KF4X creates per-namespace buckets with IAM isolation |

### 5.4 Denial of Service

| Threat | Description | Likelihood | Impact | Mitigation |
|--------|-------------|-----------|--------|------------|
| T-S3-D1 | S3 storage quota exhausted | Low | Medium | KF4X supports per-namespace S3 quotas. Monitoring recommended for bucket size |

---

## 6. Model Artefacts

### 6.1 Tampering

| Threat | Description | Likelihood | Impact | Mitigation |
|--------|-------------|-----------|--------|------------|
| T-MOD-T1 | Adversary replaces model.joblib with backdoored model | Low | Critical | Models stored in S3 with IAM controls. MLflow tracks model hash and training lineage. Model cards auto-generated on training document training data and metrics. Production should sign model artefacts |
| T-MOD-T2 | Training data poisoning produces biased model | Low | High | Pydantic schema validation filters out-of-range sensor values before training. Evidently drift detection catches distribution shifts. Model card documents bias risks |

### 6.2 Information Disclosure

| Threat | Description | Likelihood | Impact | Mitigation |
|--------|-------------|-----------|--------|------------|
| T-MOD-I1 | Model weights reverse-engineered to extract training data | Very Low | Low | RandomForest is not easily invertible. Feature importance is not sensitive (environmental parameters are public knowledge). ONNX export strips training metadata |

### 6.3 Repudiation

| Threat | Description | Likelihood | Impact | Mitigation |
|--------|-------------|-----------|--------|------------|
| T-MOD-R1 | No traceability on which model produced a prediction | Low | Medium | Prediction audit trail records model_name and model_version per inference. `PredictionResult` includes `model_strategy` field. MLflow logs training parameters and metrics |

---

## 7. Kubernetes Deployment

### 7.1 Spoofing

| Threat | Description | Likelihood | Impact | Mitigation |
|--------|-------------|-----------|--------|------------|
| T-K8S-S1 | Rogue pod impersonates ReefTwin API | Low | High | KF4X Istio `AuthorizationPolicy` restricts namespace access. Pod labels must match `app: reeftwin-api`. Network policies recommended |

### 7.2 Tampering

| Threat | Description | Likelihood | Impact | Mitigation |
|--------|-------------|-----------|--------|------------|
| T-K8S-T1 | Container image replaced with malicious version | Low | Critical | `imagePullPolicy: Always` ensures fresh pulls. Versioned image tags (not `:latest`). KF4X Kyverno add-on supports `cosign` image signature verification |

### 7.3 Denial of Service

| Threat | Description | Likelihood | Impact | Mitigation |
|--------|-------------|-----------|--------|------------|
| T-K8S-D1 | Pod consumes excessive resources, starving other workloads | Medium | Medium | Resource limits: `cpu: 1`, `memory: 1Gi`. Kubernetes `LimitRange` (KF4X) enforces namespace-level quotas |
| T-K8S-D2 | Liveness probe failure causes restart loop | Low | Medium | Liveness probe with `initialDelaySeconds: 10` and `periodSeconds: 30` prevents premature restarts. Health endpoint is lightweight |

### 7.4 Elevation of Privilege

| Threat | Description | Likelihood | Impact | Mitigation |
|--------|-------------|-----------|--------|------------|
| T-K8S-E1 | Container escapes to host | Very Low | Critical | No `privileged`, `hostNetwork`, or `hostPID` in deployment spec. `runAsNonRoot` recommended (planned). No volume mounts to host filesystem in production |

---

## 8. CI/CD Supply Chain

### 8.1 Spoofing

| Threat | Description | Likelihood | Impact | Mitigation |
|--------|-------------|-----------|--------|------------|
| T-CI-S1 | Compromised GitHub Actions runner executes malicious code | Very Low | Critical | Uses official actions (`actions/checkout@v4`, `astral-sh/setup-uv@v6`). No self-hosted runners. Repository secrets restricted to environment-scoped access |

### 8.2 Tampering

| Threat | Description | Likelihood | Impact | Mitigation |
|--------|-------------|-----------|--------|------------|
| T-CI-T1 | Dependency confusion — malicious package with same name on public registry | Low | High | `pip-audit` scans for known CVEs in CI. All dependencies pinned with minimum versions in `pyproject.toml`. No private package indices used |
| T-CI-T2 | Compromised upstream dependency introduces vulnerability | Low | High | `pip-audit` detects known CVEs. `detect-secrets` scans for leaked credentials. Pre-commit hooks run on every commit. License audit confirms all permissive (MIT/Apache/BSD) |

### 8.3 Repudiation

| Threat | Description | Likelihood | Impact | Mitigation |
|--------|-------------|-----------|--------|------------|
| T-CI-R1 | No audit trail of who deployed what | Low | Medium | Git history provides full change provenance. GitHub Actions logs all CI runs. KF4X Phase 4 ArgoCD provides GitOps audit trail |

### 8.4 Information Disclosure

| Threat | Description | Likelihood | Impact | Mitigation |
|--------|-------------|-----------|--------|------------|
| T-CI-I1 | Secrets leaked in CI logs | Low | High | Secrets stored as GitHub repository secrets (masked in logs). `.env` in `.gitignore`. `detect-secrets` scan in CI catches accidental commits |

---

## Risk Heat Map

```
           Low Impact    Medium Impact    High Impact    Critical Impact
          ┌─────────────┬────────────────┬──────────────┬───────────────┐
High      │             │ T-KAF-S1       │              │               │
Likelihood│             │ T-API-D1       │              │               │
          ├─────────────┼────────────────┼──────────────┼───────────────┤
Medium    │ T-API-S1    │ T-SIM-T2       │ T-ING-T1     │               │
          │ T-API-I1    │ T-KAF-I1       │ T-S3-S1      │               │
          │             │ T-KAF-D1       │              │               │
          │             │ T-K8S-D1       │              │               │
          ├─────────────┼────────────────┼──────────────┼───────────────┤
Low       │ T-API-R1    │ T-API-T2       │ T-S3-T2      │ T-MOD-T1      │
          │ T-SIM-I1    │ T-ING-R1       │ T-CI-T1      │ T-K8S-T1      │
          │ T-MOD-I1    │ T-MOD-R1       │ T-CI-T2      │ T-CI-S1       │
          │             │ T-CI-R1        │              │ T-K8S-E1      │
          └─────────────┴────────────────┴──────────────┴───────────────┘
```
