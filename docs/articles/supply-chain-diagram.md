# KF4X Supply Chain Security — Mermaid Diagrams

## Diagram 1: Full Supply Chain Security Pipeline

```mermaid
flowchart TB
    subgraph Sources["Public Registries"]
        direction LR
        GCR["gcr.io"]
        DOCKER["docker.io"]
        GHCR["ghcr.io"]
        QUAY["quay.io"]
        K8S["registry.k8s.io"]
    end

    subgraph PreDeploy["Pre-Deploy — Build Time"]
        direction TB
        KUSTOMIZE["Kustomize Build<br/>Extract all image references"]
        SCAN["cosign Scan<br/>install.sh"]
        REPORT["Scan Report<br/>Signed | Unsigned | Attestations"]
        KUSTOMIZE --> SCAN --> REPORT
    end

    subgraph Mirror["Registry Mirror — Kyverno Mutating Webhook"]
        direction TB
        REWRITE["ClusterPolicy: Image Rewrite<br/>gcr.io/... → mirror/gcr.io/..."]
        PRIVATE["Private Registry<br/>Air-gapped / Enterprise"]
        REWRITE --> PRIVATE
    end

    subgraph Admission["Deploy Time — Kyverno Admission Controller"]
        direction TB
        POD["Pod Creation<br/>Kubernetes API"]
        VERIFY{"Verify cosign<br/>Signature"}
        SIGNED["Signed<br/>Policy passes"]
        UNSIGNED["Unsigned<br/>Audit logged"]
        DOWN["Kyverno Down<br/>Fail-open"]
        POD --> VERIFY
        VERIFY -->|"Valid signature"| SIGNED
        VERIFY -->|"Missing / invalid"| UNSIGNED
        VERIFY -->|"Kyverno unavailable"| DOWN
    end

    subgraph Runtime["Runtime — What Gets Protected"]
        direction LR
        NB["Notebooks"]
        PIPE["Pipelines"]
        SERVE["KServe Models"]
        SPARK["Spark Jobs"]
        OPS["Operators"]
    end

    Sources --> PreDeploy
    Sources --> Mirror
    Mirror --> Admission
    PreDeploy -.->|"Report informs<br/>security posture"| Admission
    SIGNED --> Runtime
    UNSIGNED -->|"Pod runs<br/>violation recorded"| Runtime
    DOWN -->|"Pod runs<br/>availability preserved"| Runtime

    style Sources fill:#fff3e0,stroke:#e65100
    style PreDeploy fill:#e3f2fd,stroke:#1565c0
    style Mirror fill:#f3e5f5,stroke:#6a1b9a
    style Admission fill:#e8f5e9,stroke:#2e7d32
    style Runtime fill:#fff9c4,stroke:#f9a825
    style SIGNED fill:#c8e6c9,stroke:#2e7d32
    style UNSIGNED fill:#fff9c4,stroke:#f9a825
    style DOWN fill:#ffcdd2,stroke:#c62828
```

## Diagram 2: Admission Decision Flow (matches kyverno-policy.png)

```mermaid
flowchart LR
    POD["Pod Creation<br/>Kubernetes API<br/>(kube-apiserver)"]

    KYVERNO{"Kyverno<br/>Admission<br/>Verify image signature"}

    ALLOWED["✅ Allowed<br/>Pod is allowed to run"]
    AUDIT["⚠️ Audit Log<br/>Violation logged.<br/>Pod still runs."]
    FAILOPEN["🔄 Fail-Open<br/>Kyverno unavailable.<br/>Pod still runs."]

    POD --> KYVERNO
    KYVERNO -->|"Signed"| ALLOWED
    KYVERNO -->|"Unsigned"| AUDIT
    KYVERNO -->|"Kyverno down"| FAILOPEN

    style ALLOWED fill:#c8e6c9,stroke:#2e7d32,color:#1b5e20
    style AUDIT fill:#fff9c4,stroke:#f9a825,color:#e65100
    style FAILOPEN fill:#ffcdd2,stroke:#c62828,color:#b71c1c
    style KYVERNO fill:#e8f5e9,stroke:#2e7d32
```

## Diagram 3: Defense-in-Depth Layers

```mermaid
flowchart TB
    subgraph L1["Layer 1 — Source Control"]
        MIRROR["Registry Mirroring<br/>Kyverno rewrites all image refs<br/>to private registry"]
    end

    subgraph L2["Layer 2 — Build-Time Scanning"]
        COSIGN["cosign Scan Report<br/>Every image checked for<br/>signatures + attestations"]
    end

    subgraph L3["Layer 3 — Admission Verification"]
        KYVERNO["Kyverno Admission<br/>Verify cosign signatures<br/>at pod creation"]
    end

    subgraph L4["Layer 4 — Audit Trail"]
        AUDIT["Policy Violations<br/>Logged, not blocked<br/>Security team has visibility"]
    end

    subgraph L5["Layer 5 — Availability Guarantee"]
        FAILOPEN["Fail-Open Design<br/>Workloads never blocked<br/>by policy engine failure"]
    end

    L1 --> L2 --> L3 --> L4 --> L5

    style L1 fill:#e3f2fd,stroke:#1565c0
    style L2 fill:#f3e5f5,stroke:#6a1b9a
    style L3 fill:#e8f5e9,stroke:#2e7d32
    style L4 fill:#fff3e0,stroke:#e65100
    style L5 fill:#fce4ec,stroke:#c62828
```
