# ReefTwin Risk Register

> **Purpose:** Tracks identified security and AI risks, their current treatment status, and responsible parties.
> **Methodology:** Derived from STRIDE threat model, security-aligned with ASD Essential Eight, OWASP API Security Top 10 (2023), CIS Kubernetes Benchmark, and NIST AI RMF 1.0.
> **Version:** 0.1.0
> **Last reviewed:** 2026-05-07
> **Review cadence:** Quarterly or after significant architecture changes

---

## Risk Rating Criteria

**Likelihood:**
- **Rare** — requires advanced attacker with internal knowledge and sustained access
- **Unlikely** — requires specific conditions or elevated access
- **Possible** — feasible with moderate effort and publicly available tools
- **Likely** — low barrier, common attack vector

**Impact:**
- **Minor** — no data loss, minor inconvenience, self-recovering
- **Moderate** — service degradation, delayed data refresh, limited data exposure
- **Major** — incorrect predictions relied upon for decisions, data integrity compromised
- **Severe** — complete system compromise, model supply chain poisoned, credential exfiltration

**Risk Level** = Likelihood x Impact

| | Minor | Moderate | Major | Severe |
|---|---|---|---|---|
| **Likely** | Medium | High | Critical | Critical |
| **Possible** | Low | Medium | High | Critical |
| **Unlikely** | Low | Low | Medium | High |
| **Rare** | Low | Low | Low | Medium |

---

## Active Risks

### R-001: Unauthenticated Access to LLM Endpoints

| Field | Value |
|-------|-------|
| **Risk ID** | R-001 |
| **Threat Ref** | T-API-S1, T-API-S2 |
| **Category** | Spoofing |
| **Description** | LLM endpoints (`/rag`, `/agent`, `/interpret`) consume paid API credits. Unauthenticated access allows cost abuse |
| **Likelihood** | Possible |
| **Impact** | Moderate (financial cost, not data loss) |
| **Risk Level** | **Medium** |
| **Current Controls** | API key auth (`X-API-Key`), rate limiter (30 req/min), mock fallback when no LLM key |
| **Residual Risk** | Low — rate limiter caps maximum cost exposure |
| **Treatment** | Accept with monitoring. Track `rag_queries_total` and `agent_queries_total` Prometheus counters |
| **Owner** | Platform team |
| **Status** | Mitigated |

---

### R-002: Model Artefact Tampering

| Field | Value |
|-------|-------|
| **Risk ID** | R-002 |
| **Threat Ref** | T-MOD-T1, T-S3-T1 |
| **Category** | Tampering |
| **Description** | Adversary replaces `model.joblib` or ONNX artefact with a backdoored model that produces intentionally misleading bleaching risk predictions |
| **Likelihood** | Unlikely |
| **Impact** | Severe (incorrect predictions could influence reef management decisions) |
| **Risk Level** | **High** |
| **Current Controls** | S3 IAM-scoped credentials, MLflow model versioning, auto-generated model cards with training metadata, prediction audit trail |
| **Residual Risk** | Medium — no cryptographic model signing yet |
| **Treatment** | Reduce. Implement model artefact signing (cosign/sigstore). Enable S3 versioning with MFA delete |
| **Owner** | ML Engineering |
| **Status** | Partially mitigated — signing planned |

---

### R-003: Training Data Poisoning

| Field | Value |
|-------|-------|
| **Risk ID** | R-003 |
| **Threat Ref** | T-MOD-T2, T-ING-T1 |
| **Category** | Tampering |
| **Description** | Corrupted or adversarial data enters the feature pipeline, biasing model training. Could produce systematically incorrect bleaching predictions |
| **Likelihood** | Unlikely |
| **Impact** | Major |
| **Risk Level** | **Medium** |
| **Current Controls** | Pydantic schema validation (range checks on all sensor fields), DLQ for rejected records, Evidently drift detection, NOAA URL hardcoded (not user-supplied) |
| **Residual Risk** | Low — schema validation catches out-of-range values; drift detection catches distribution shifts |
| **Treatment** | Accept with monitoring. Review DLQ contents quarterly. Retrain model if Evidently drift score exceeds critical threshold |
| **Owner** | Data Engineering |
| **Status** | Mitigated |

---

### R-004: Kafka/Redpanda Unauthorised Access

| Field | Value |
|-------|-------|
| **Risk ID** | R-004 |
| **Threat Ref** | T-KAF-S1, T-KAF-I1 |
| **Category** | Spoofing, Information Disclosure |
| **Description** | Redpanda broker and Console accessible without authentication in development configuration. Unauthorised producer could inject malicious sensor events |
| **Likelihood** | Possible (in dev), Unlikely (in prod with ACLs) |
| **Impact** | Major (data integrity) |
| **Risk Level** | **High** (dev), **Medium** (prod) |
| **Current Controls** | Internal port (9092) not exposed externally. External port (19092) is dev-only. Schema validation at consumer rejects invalid messages |
| **Residual Risk** | Medium in dev environments |
| **Treatment** | Reduce. Enable Redpanda SASL/SCRAM authentication and TLS in production. Remove Console from production compose |
| **Owner** | Platform team |
| **Status** | Partially mitigated — auth planned for production |

---

### R-005: S3 Credential Leakage

| Field | Value |
|-------|-------|
| **Risk ID** | R-005 |
| **Threat Ref** | T-S3-S1, T-CI-I1 |
| **Category** | Information Disclosure |
| **Description** | S3 access key and secret key exposed via CI logs, error messages, or accidentally committed to source control |
| **Likelihood** | Unlikely |
| **Impact** | Severe (full data access) |
| **Risk Level** | **High** |
| **Current Controls** | Credentials in env vars only (not source control). `.env` in `.gitignore`. `detect-secrets` scan in CI. GitHub repository secrets masked in logs |
| **Residual Risk** | Low |
| **Treatment** | Accept. Rotate credentials quarterly. Use AWS IAM roles (no long-lived keys) for SageMaker deployments |
| **Owner** | Security |
| **Status** | Mitigated |

---

### R-006: Prompt Injection via LLM Endpoints

| Field | Value |
|-------|-------|
| **Risk ID** | R-006 |
| **Threat Ref** | T-SIM-T2 |
| **Category** | Tampering |
| **Description** | User-supplied query to `/rag` or `/agent` contains prompt injection that causes the LLM to bypass system instructions, execute unintended tool calls, or leak system prompt content |
| **Likelihood** | Possible |
| **Impact** | Moderate (incorrect answers, unintended simulations — no data mutation via tools) |
| **Risk Level** | **Medium** |
| **Current Controls** | Agent tools are read-only (query_reef_state, search_knowledge_base) or bounded (run_simulation validates parameters server-side). Query length limited to 2000 chars. Rate limiter bounds total calls |
| **Residual Risk** | Low — tools cannot modify data; worst case is incorrect answers |
| **Treatment** | Accept. Agent tools are deliberately read-only. Monitor tool call patterns via `agent_tool_calls_total` Prometheus metric |
| **Owner** | ML Engineering |
| **Status** | Mitigated |

---

### R-007: Container Image Supply Chain Compromise

| Field | Value |
|-------|-------|
| **Risk ID** | R-007 |
| **Threat Ref** | T-K8S-T1, T-CI-T1, T-CI-T2 |
| **Category** | Tampering |
| **Description** | Compromised base image or upstream Python dependency introduces malicious code into production containers |
| **Likelihood** | Rare |
| **Impact** | Severe |
| **Risk Level** | **Medium** |
| **Current Controls** | Multi-stage Dockerfile with pinned base (`python:3.12-slim`). `pip-audit` CVE scan in CI. `detect-secrets` scan. Pre-commit hooks. All dependencies permissive-licensed (MIT/Apache/BSD) |
| **Residual Risk** | Low |
| **Treatment** | Reduce. Add SBOM generation (`syft`/`cdxgen`). Enable Kyverno image signing verification. Pin dependency hashes (`uv pip compile --generate-hashes`) |
| **Owner** | Platform team |
| **Status** | Partially mitigated — SBOM and signing planned |

---

### R-008: Kubernetes Privilege Escalation

| Field | Value |
|-------|-------|
| **Risk ID** | R-008 |
| **Threat Ref** | T-K8S-E1 |
| **Category** | Elevation of Privilege |
| **Description** | Container process escapes to host or accesses cluster resources beyond its namespace |
| **Likelihood** | Rare |
| **Impact** | Severe |
| **Risk Level** | **Medium** |
| **Current Controls** | No `privileged`, `hostNetwork`, or `hostPID` in deployment. Resource limits set. KF4X provides per-namespace isolation via Istio `AuthorizationPolicy` |
| **Residual Risk** | Low |
| **Treatment** | Reduce. Add `securityContext.runAsNonRoot: true` and `readOnlyRootFilesystem: true`. Add `NetworkPolicy` resources |
| **Owner** | Platform team |
| **Status** | Partially mitigated — securityContext planned |

---

### R-009: AI Model Bias Affecting Reef Management Decisions

| Field | Value |
|-------|-------|
| **Risk ID** | R-009 |
| **Threat Ref** | NIST AI RMF MAP function |
| **Category** | AI Risk (not STRIDE) |
| **Description** | Bleaching risk model trained on GBR-biased synthetic data may not generalise to other reef regions. False negatives could delay protective interventions |
| **Likelihood** | Possible |
| **Impact** | Major (ecological impact if used for real decisions) |
| **Risk Level** | **High** |
| **Current Controls** | Model card documents: "Not suitable for real conservation decisions without validation." Bias risks enumerated (GBR training bias, synthetic data limitations, threshold assumptions). Conformal prediction provides uncertainty intervals |
| **Residual Risk** | Medium — model is portfolio/research prototype, not production decision system |
| **Treatment** | Accept for portfolio use. Require validation on AIMS/NOAA verified datasets before any real-world deployment. Add fairness metrics (SHAP, demographic parity across reef regions) |
| **Owner** | ML Engineering |
| **Status** | Documented — formal validation required before real-world use |

---

### R-010: External API Dependency Failure

| Field | Value |
|-------|-------|
| **Risk ID** | R-010 |
| **Threat Ref** | T-ING-D1 |
| **Category** | Denial of Service |
| **Description** | NOAA CoastWatch ERDDAP API or Anthropic/OpenAI LLM API unavailable, degrading platform functionality |
| **Likelihood** | Possible |
| **Impact** | Moderate (degraded, not broken — fallbacks exist) |
| **Risk Level** | **Medium** |
| **Current Controls** | NOAA: automatic fallback to synthetic data. LLM: mock provider fallback (all GenAI features return placeholder responses). Inference: fallback chain (RF → PIML → heuristic) |
| **Residual Risk** | Low — platform remains functional in degraded mode |
| **Treatment** | Accept. Graceful degradation by design. Monitor API availability via health checks |
| **Owner** | Platform team |
| **Status** | Mitigated |

---

## Risk Summary

| Risk Level | Count | Risk IDs |
|-----------|-------|----------|
| **Critical** | 0 | — |
| **High** | 3 | R-002, R-004, R-005, R-009 |
| **Medium** | 6 | R-001, R-003, R-006, R-007, R-008, R-010 |
| **Low** | 0 | — |

---

## Treatment Plan Priority

| Priority | Action | Risk(s) Addressed | Effort | Status |
|----------|--------|-------------------|--------|--------|
| 1 | `securityContext.runAsNonRoot` + `readOnlyRootFilesystem` + `drop ALL` caps | R-008 | 10 min | **Done** |
| 2 | Redpanda SASL auth in `docker-compose.prod.yml` + no Console | R-004 | 30 min | **Done** |
| 3 | Model signing script (`cosign`) + `verify_model_integrity()` SHA256 hash | R-002 | 2 hrs | **Done** |
| 4 | SBOM generation (`anchore/sbom-action`) in CI | R-007 | 30 min | **Done** |
| 5 | Dependency hashes in `requirements.lock` | R-007 | 15 min | **Done** |
| 6 | S3 versioning auto-enabled on bucket creation | R-002, R-005 | 30 min | **Done** |
| 7 | AI fairness: permutation importance + group parity across reefs | R-009 | 2 hrs | **Done** |
| 8 | OAuth2/OIDC JWT auth (`require_oidc_token`) for Keycloak/Auth0/Entra ID | R-001 | 4 hrs | **Done** |

---

## Appendix: Framework Cross-Reference

| Risk | ASD E8 | OWASP | CIS K8s | NIST AI RMF |
|------|--------|-------|---------|-------------|
| R-001 | E5 (Restrict admin), E7 (MFA) | API2 (Broken auth) | — | — |
| R-002 | E1 (App control) | — | 5.7 (Images) | MANAGE (model governance) |
| R-003 | E4 (User app hardening) | API10 (Unsafe consumption) | — | MAP (bias), MEASURE (drift) |
| R-004 | E5 (Restrict admin) | API2 (Broken auth) | — | — |
| R-005 | E5 (Restrict admin) | API8 (Security misconfig) | 5.4 (Secrets) | — |
| R-006 | E4 (User app hardening) | API4 (Unrestricted consumption) | — | MANAGE (incident response) |
| R-007 | E1 (App control), E2 (Patch apps) | — | 5.7 (Images) | — |
| R-008 | E5 (Restrict admin) | — | 5.1 (RBAC), 5.2 (Pod security) | — |
| R-009 | — | — | — | MAP (bias), MEASURE (performance), GOVERN (policy) |
| R-010 | E8 (Backups) | API6 (Sensitive flows) | — | MANAGE (resilience) |
