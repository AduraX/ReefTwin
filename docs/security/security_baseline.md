# ReefTwin Security Baseline

> **Status:** Security-aligned with ASD Essential Eight, ASD ISM, OWASP API Security Top 10 (2023), CIS Kubernetes Benchmark, and NIST AI RMF 1.0. This document describes the security posture of the ReefTwin platform. It does not claim full compliance with any framework — it describes alignment with their principles and controls.

**Version:** 0.1.0
**Last reviewed:** 2026-05-07
**Owner:** ReefTwin Engineering

---

## 1. ASD Essential Eight Alignment

The Australian Signals Directorate Essential Eight provides a baseline of mitigation strategies against cyber threats. ReefTwin aligns its practices to these eight areas.

### E1: Application Control

| Control | ReefTwin Implementation | Maturity |
|---------|------------------------|----------|
| Restrict execution to approved applications | Docker images built from explicit `Dockerfile` with pinned base images (`python:3.12-slim`). No arbitrary code execution in production containers | Developing |
| Container image provenance | Kyverno image verification available via KF4X Phase 1 add-on. `cosign` signature checking supported | Planned |
| CI artefact integrity | GitHub Actions CI builds from source; no pre-built binary downloads in pipeline | Implemented |

### E2: Patch Applications

| Control | ReefTwin Implementation | Maturity |
|---------|------------------------|----------|
| Dependency vulnerability scanning | `pip-audit` runs in CI security job; alerts on known CVEs | Implemented |
| Automated dependency updates | Dependabot or Renovate recommended for GitHub repository | Planned |
| Base image updates | Docker images use `python:3.12-slim`; CI rebuilds weekly recommended | Planned |

### E3: Configure Microsoft Office Macros

Not applicable — ReefTwin is a server-side platform with no Office macro execution.

### E4: User Application Hardening

| Control | ReefTwin Implementation | Maturity |
|---------|------------------------|----------|
| API input validation | Pydantic schema validation on all POST endpoints. `reef_id` format enforcement (alphanumeric + underscore, max 64 chars). Query length bounds (max 2000 chars) | Implemented |
| Content-Type enforcement | FastAPI enforces `application/json` on all POST routes | Implemented |
| Error message sanitisation | FastAPI production mode does not expose stack traces. Custom `HTTPException` messages used | Implemented |

### E5: Restrict Administrative Privileges

| Control | ReefTwin Implementation | Maturity |
|---------|------------------------|----------|
| API authentication | `X-API-Key` header authentication on LLM endpoints (`/rag`, `/agent`). Configurable via `REEFTWIN_API_KEYS` env var. Disabled in dev (no key = open) | Implemented |
| Kubernetes RBAC | KF4X Phase 1 provides per-namespace RBAC via Keycloak groups. Kubeflow profiles enforce `kubeflow-edit` / `kubeflow-view` roles | Delegated to KF4X |
| S3 bucket isolation | KF4X provides per-namespace IAM identities and scoped IAM policies for SeaweedFS buckets | Delegated to KF4X |
| Principle of least privilege | API key grants access to all authenticated endpoints (no role-based granularity yet) | Developing |

### E6: Patch Operating Systems

| Control | ReefTwin Implementation | Maturity |
|---------|------------------------|----------|
| Container OS patching | Base image `python:3.12-slim` is Debian-based; regular rebuilds pull latest security patches | Planned |
| Host OS patching | Delegated to Kubernetes cluster operator (KF4X, EKS, GKE, etc.) | Out of scope |

### E7: Multi-Factor Authentication

| Control | ReefTwin Implementation | Maturity |
|---------|------------------------|----------|
| API authentication | API key-based (single factor). MFA not enforced at API level | Developing |
| Dashboard authentication | When deployed on KF4X, Keycloak provides MFA (email, SMS, TOTP) for Streamlit/React access via oauth2-proxy | Delegated to KF4X |
| CI/CD access | GitHub Actions uses repository secrets; GitHub account MFA recommended | Recommended |

### E8: Regular Backups

| Control | ReefTwin Implementation | Maturity |
|---------|------------------------|----------|
| Data layer backups | S3 data store supports versioned buckets (SeaweedFS, AWS S3). Iceberg tables (KF4X Phase 3) provide time-travel for data recovery | Planned |
| Model artefact backups | MLflow Model Registry versions all model artefacts on SeaweedFS S3 | Implemented |
| Configuration backups | All configuration in version control (git). `.env` excluded from repository | Implemented |

---

## 2. ASD ISM-Style Controls

The following controls reflect ISM (Information Security Manual) thinking applied to the ReefTwin context. ISM control IDs are referenced for traceability where applicable.

### Access Control (ISM-0432, ISM-1503)

- API endpoints require `X-API-Key` for LLM-consuming routes
- Kubernetes deployment uses `readinessProbe` and `livenessProbe` to prevent routing to unhealthy instances
- No default credentials shipped; all secrets via environment variables
- Dashboard access controlled by Keycloak on KF4X deployments

### Cryptographic Controls (ISM-0457, ISM-1139)

- HTTPS/TLS termination delegated to ingress controller (Traefik/Nginx) or load balancer
- S3 credentials passed via environment variables, not configuration files
- No cryptographic material stored in source control (`.gitignore` excludes `.env`, `.key`, `.pem`)
- API keys compared via constant-time string comparison recommended (not yet enforced)

### Network Security (ISM-1416, ISM-0535)

- Docker Compose exposes only necessary ports (8000 API, 3000 Grafana, 9090 Prometheus)
- Kubernetes `Service` defaults to `ClusterIP` (no external exposure without explicit Ingress)
- KF4X provides Istio service mesh with `AuthorizationPolicy` enforcement per namespace
- Redpanda (Kafka) internal port (9092) not exposed externally; external access on 19092 for development only

### Logging and Monitoring (ISM-0580, ISM-0585)

- Structured logging via Python `logging` module (`reeftwin.*` namespace)
- 8 Prometheus metrics exported at `/metrics` endpoint
- Grafana dashboard provisioned with 8 panels (latency, rates, drift, tools)
- Prediction audit trail: append-only JSONL at `data/audit/predictions.jsonl`
- MLflow tracing on inference calls logs latency per prediction
- Evidently AI drift reports generate timestamped HTML artefacts

### Data Classification

| Classification | Data | Controls |
|---------------|------|----------|
| **OFFICIAL** | Reef state JSON, feature CSVs | S3 bucket ACLs, per-namespace isolation |
| **OFFICIAL: Sensitive** | API keys, S3 credentials | Environment variables only; never in source control |
| **Not classified** | Synthetic/sample data, model weights | No sensitivity restrictions |

---

## 3. OWASP API Security Top 10 (2023) Alignment

| # | OWASP Risk | ReefTwin Status | Implementation |
|---|-----------|-----------------|----------------|
| API1 | Broken Object Level Authorization | Partially mitigated | `reef_id` validated for format but no ownership checks (any authenticated user can query any reef). Acceptable for decision-support platform — not multi-tenant user data | 
| API2 | Broken Authentication | Mitigated | `X-API-Key` header on LLM endpoints. Mock provider falls back gracefully without key. Keys loaded from `REEFTWIN_API_KEYS` env var |
| API3 | Broken Object Property Level Authorization | Low risk | API returns full reef state objects. No partial field filtering. Acceptable — reef data is not PII |
| API4 | Unrestricted Resource Consumption | Mitigated | Rate limiter (30 req/min) on `/rag`, `/agent`, `/interpret`. Pydantic validates `duration_days` range (1-365). Query length capped at 2000 chars |
| API5 | Broken Function Level Authorization | Partially mitigated | All endpoints share single API key tier. No role-based access (e.g., read-only vs admin). Acceptable for v0.1 |
| API6 | Unrestricted Access to Sensitive Business Flows | Mitigated | Simulation and agent endpoints rate-limited. LLM calls cost-controlled via cache (95.5% hit rate reduces invocations) |
| API7 | Server Side Request Forgery | Low risk | NOAA API integration uses hardcoded CoastWatch ERDDAP URL. No user-supplied URLs passed to server-side requests |
| API8 | Security Misconfiguration | Mitigated | CORS restricted to `http://localhost:3001` (React dev). Production should use explicit origin allowlist. Swagger docs at `/docs` disabled via env var in production recommended |
| API9 | Improper Inventory Management | Mitigated | All 9 API endpoints documented in README. Swagger UI auto-generates from Pydantic schemas. No shadow APIs |
| API10 | Unsafe Consumption of APIs | Partially mitigated | NOAA API responses validated via column mapping. LLM responses not sanitised before display (XSS risk in dashboard if content rendered as HTML — Streamlit escapes by default, React escapes by default via JSX) |

---

## 4. CIS Kubernetes Benchmark Alignment

Applicable when deployed on Kubernetes (standalone or KF4X).

| CIS Control Area | ReefTwin Implementation | Status |
|-------------------|------------------------|--------|
| **1.1 API Server** | Delegated to cluster operator. KF4X enforces Istio `RequestAuthentication` with Keycloak JWT validation | Delegated |
| **3.2 Logging** | Application logs to stdout (captured by cluster logging). Prometheus metrics exposed | Implemented |
| **4.1 Worker node security** | Delegated to cluster operator | Out of scope |
| **5.1 RBAC** | `deployment.yaml` does not use `hostNetwork`, `hostPID`, or `privileged`. No `runAsRoot` required | Implemented |
| **5.2 Pod Security** | Resource limits set: `cpu: 250m-1`, `memory: 512Mi-1Gi`. No privilege escalation capabilities requested | Implemented |
| **5.3 Network Policies** | KF4X provides Kyverno-based network policies per namespace (add-on). Standalone deployment should add `NetworkPolicy` resources | Planned |
| **5.4 Secrets Management** | Secrets via environment variables, not mounted files. KF4X uses Kubernetes Secrets with `PodDefault` injection | Implemented |
| **5.7 Container Images** | Multi-stage Dockerfile with slim base. `imagePullPolicy: Always` in deployment. Versioned tags (not `:latest`) | Implemented |

---

## 5. NIST AI RMF 1.0 Alignment

The NIST AI Risk Management Framework (AI RMF 1.0) provides guidance for trustworthy AI. ReefTwin aligns with its four core functions.

### GOVERN (AI Governance)

| Practice | ReefTwin Implementation |
|----------|------------------------|
| AI risk management policy | Model cards document intended use, limitations, bias risks, and ethical considerations for each model |
| Roles and responsibilities | `CONTRIBUTING.md` defines development process. Model card `owner` field identifies responsible party |
| Documentation standards | Standardised `ModelCard` dataclass with required fields: `intended_use`, `limitations`, `ethical_considerations`, `bias_risks` |
| Lifecycle management | MLflow Model Registry tracks model versions with aliases (`staging`, `production`) |

### MAP (Risk Identification)

| Practice | ReefTwin Implementation |
|----------|------------------------|
| Context and use case | Model cards specify "Portfolio demonstration and research prototype. Not suitable for real conservation decisions without validation" |
| Data provenance | Data lineage graph (`DataLineage` class) traces source → transform → model → prediction |
| Bias identification | Model card `bias_risks` field documents: GBR training bias, synthetic data limitations, threshold assumptions |
| Stakeholder impact | Model card notes: "False negatives (missed bleaching) could delay protective interventions" |

### MEASURE (Risk Assessment)

| Practice | ReefTwin Implementation |
|----------|------------------------|
| Performance measurement | MLflow tracks ROC-AUC, F1, precision, recall per training run |
| Uncertainty quantification | Conformal prediction provides calibrated 90% prediction intervals |
| Drift detection | Evidently AI (K-S test, Wasserstein) + custom PSI monitor detect distribution shifts |
| Benchmark experiments | Three experiments with before/after measurements: latency, cost, reliability |

### MANAGE (Risk Treatment)

| Practice | ReefTwin Implementation |
|----------|------------------------|
| Prediction audit trail | Append-only JSONL log records: timestamp, reef_id, model used, prediction, input features, latency |
| Fallback mechanisms | `FallbackChain` tries RF → PIML → heuristic. `HeuristicPredictor` requires no trained model |
| Model governance | Auto-generated model cards on training. Governance metadata logged to MLflow |
| Incident response | DLQ captures failed records for post-mortem. Drift alerts trigger via Prometheus `drift_alerts_total` counter |

---

## 6. Security Controls Summary

| Domain | Controls Implemented | Controls Planned |
|--------|---------------------|-----------------|
| **Authentication** | API key auth on LLM endpoints | Role-based access, OAuth2/OIDC integration |
| **Authorisation** | Single-tier API key | Per-endpoint RBAC, reef-level ACLs |
| **Input validation** | Pydantic schemas, reef_id format, query length | Request signing, HMAC verification |
| **Rate limiting** | 30 req/min on LLM endpoints | Per-user quotas, token-based billing |
| **Encryption** | TLS at ingress (delegated) | At-rest encryption for S3 buckets |
| **Logging** | Structured logs, audit trail, MLflow tracing | SIEM integration, log correlation IDs |
| **Monitoring** | 8 Prometheus metrics, Grafana dashboard | Alerting rules, PagerDuty integration |
| **Supply chain** | `pip-audit` in CI, `detect-secrets` scan | SBOM generation, image signing |
| **Container security** | Multi-stage build, slim base, resource limits | Read-only filesystem, non-root user |
| **AI governance** | Model cards, data lineage, prediction audit | Fairness metrics, explainability reports |
| **Data protection** | S3 bucket isolation, env var secrets | Encryption at rest, data retention policy |
| **Resilience** | Fallback chains, DLQ, retries | Circuit breakers, chaos testing |

---

## 7. Improvement Roadmap

### Short-term (next release)

1. Add `securityContext.runAsNonRoot: true` to Kubernetes deployment
2. Implement correlation IDs across logs and audit trail
3. Add SBOM generation (`syft` or `cdxgen`) to CI
4. Configure Prometheus alerting rules for drift and error rate thresholds

### Medium-term

5. Integrate OAuth2/OIDC for API authentication (align with KF4X Keycloak)
6. Add per-endpoint role-based access control
7. Implement at-rest encryption for S3 data layer
8. Add network policies to Kubernetes deployment

### Long-term

9. SOC 2 Type II alignment for production deployment
10. Formal penetration testing
11. AI fairness and explainability framework (SHAP, LIME integration)
12. Formal data governance policy with retention schedules
