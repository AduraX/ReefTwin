"""AI governance: model cards, audit trails, and data lineage.

Implements responsible AI practices required by AIMS JD:
  - Model cards with standardized metadata
  - Prediction audit trail (who/when/what/why)
  - Data lineage tracking (source → processing → model → prediction)

Addresses JD Responsibility 11: "Champion best practices in AI/ML
operationalisation, model governance, ethical AI, and reproducible
research-to-production workflows."
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from infrastructure.logging import get_logger

logger = get_logger("mlops.governance")


# ---------------------------------------------------------------------------
# Model Card
# ---------------------------------------------------------------------------

@dataclass
class ModelCard:
    """Standardized model documentation following ML model card practices."""

    model_name: str
    version: str
    model_type: str
    description: str

    # Training details
    training_data: str = ""
    training_date: str = ""
    features: list[str] = field(default_factory=list)
    target: str = ""
    hyperparameters: dict[str, Any] = field(default_factory=dict)

    # Performance
    metrics: dict[str, float] = field(default_factory=dict)
    evaluation_data: str = ""

    # Responsible AI
    intended_use: str = ""
    limitations: str = ""
    ethical_considerations: str = ""
    fairness_analysis: str = ""
    bias_risks: list[str] = field(default_factory=list)

    # Lineage
    data_sources: list[str] = field(default_factory=list)
    preprocessing_steps: list[str] = field(default_factory=list)
    dependencies: list[str] = field(default_factory=list)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2))
        logger.info("Model card saved: %s → %s", self.model_name, path)

    @classmethod
    def load(cls, path: str | Path) -> ModelCard:
        return cls(**json.loads(Path(path).read_text()))

    def to_markdown(self) -> str:
        lines = [
            f"# Model Card: {self.model_name} v{self.version}",
            "",
            f"**Type:** {self.model_type}",
            f"**Description:** {self.description}",
            "",
            "## Training",
            f"- **Data:** {self.training_data}",
            f"- **Date:** {self.training_date}",
            f"- **Features:** {', '.join(self.features)}",
            f"- **Target:** {self.target}",
        ]
        if self.hyperparameters:
            lines.append(f"- **Hyperparameters:** {json.dumps(self.hyperparameters)}")
        lines += [
            "",
            "## Performance",
        ]
        for k, v in self.metrics.items():
            lines.append(f"- **{k}:** {v:.4f}")
        lines += [
            "",
            "## Intended Use",
            self.intended_use or "Not specified.",
            "",
            "## Limitations",
            self.limitations or "Not specified.",
            "",
            "## Ethical Considerations",
            self.ethical_considerations or "Not specified.",
        ]
        if self.bias_risks:
            lines += ["", "## Bias Risks"]
            for risk in self.bias_risks:
                lines.append(f"- {risk}")
        if self.data_sources:
            lines += ["", "## Data Sources"]
            for src in self.data_sources:
                lines.append(f"- {src}")
        return "\n".join(lines)


def create_bleaching_model_card(metrics: dict[str, Any], version: str = "0.1.0") -> ModelCard:
    """Create a model card for the bleaching risk model."""
    return ModelCard(
        model_name="ReefTwin Bleaching Risk Model",
        version=version,
        model_type="RandomForestClassifier (scikit-learn Pipeline with StandardScaler)",
        description=(
            "Predicts near-term coral bleaching risk for a reef location using "
            "environmental and heat-stress features. Outputs a risk score (0-1) "
            "and categorical risk level (normal/watch/warning/alert)."
        ),
        training_data="Synthetic IoT sensor readings + NOAA-style heat-stress data",
        training_date=datetime.now(timezone.utc).isoformat(),
        features=[
            "water_temperature_c", "ph", "salinity_psu", "turbidity_ntu",
            "dissolved_oxygen_mg_l", "sst_anomaly_c", "hotspot_c",
            "degree_heating_weeks", "temperature_trend_7d",
        ],
        target="bleaching_label",
        hyperparameters={"n_estimators": 120, "class_weight": "balanced", "random_state": 42},
        metrics={k: v for k, v in metrics.items() if isinstance(v, (int, float)) and v is not None},
        intended_use=(
            "Portfolio demonstration and research prototype. Decision support for "
            "reef managers monitoring bleaching risk across GBR reef sites."
        ),
        limitations=(
            "Trained on synthetic data — not validated on real reef monitoring datasets. "
            "Predictions should not be used for real conservation decisions without "
            "validation against AIMS/NOAA verified observations."
        ),
        ethical_considerations=(
            "Model predictions could influence reef management resource allocation. "
            "False negatives (missed bleaching) could delay protective interventions. "
            "Model should be used as one input alongside expert judgment."
        ),
        bias_risks=[
            "Training data biased toward GBR reef conditions — may not generalize to other regions",
            "Synthetic data may not capture real-world sensor noise and failure modes",
            "Heat-stress thresholds based on published literature — may not reflect local adaptation",
        ],
        data_sources=[
            "Simulated IoT reef sensors (temperature, pH, salinity, turbidity, DO)",
            "NOAA Coral Reef Watch (SST, HotSpot, DHW, Bleaching Alert Area) — sample data",
        ],
        preprocessing_steps=[
            "IoT readings aggregated to daily means per reef",
            "NOAA data merged by reef_id and date",
            "Missing values: forward-fill within reef, then column medians",
            "7-day temperature trend computed via rolling window",
            "Bleaching label derived from DHW >= 4 OR temp >= 30°C OR hotspot >= 1°C",
        ],
        dependencies=["scikit-learn>=1.5.0", "pandas>=2.2.0", "numpy>=1.26.0", "joblib>=1.4.0"],
    )


# ---------------------------------------------------------------------------
# Audit Trail
# ---------------------------------------------------------------------------

@dataclass
class AuditEntry:
    """Single entry in the prediction audit trail."""

    timestamp: str
    reef_id: str
    model_name: str
    model_version: str
    prediction: dict[str, Any]
    input_features: dict[str, float]
    latency_ms: float
    provider: str = ""  # LLM provider if GenAI involved


class AuditTrail:
    """Append-only prediction audit trail for AI governance."""

    def __init__(self, path: str | Path = "data/audit/predictions.jsonl") -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, entry: AuditEntry) -> None:
        with open(self.path, "a") as f:
            f.write(json.dumps(asdict(entry)) + "\n")

    def read_all(self) -> list[AuditEntry]:
        if not self.path.exists():
            return []
        entries = []
        for line in self.path.read_text().strip().split("\n"):
            if line:
                entries.append(AuditEntry(**json.loads(line)))
        return entries

    @property
    def count(self) -> int:
        if not self.path.exists():
            return 0
        return sum(1 for line in self.path.read_text().strip().split("\n") if line)


# ---------------------------------------------------------------------------
# Data Lineage
# ---------------------------------------------------------------------------

@dataclass
class LineageNode:
    name: str
    node_type: str  # "source", "transform", "model", "output"
    inputs: list[str] = field(default_factory=list)
    outputs: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


class DataLineage:
    """Tracks data flow from source through processing to predictions."""

    def __init__(self) -> None:
        self._nodes: dict[str, LineageNode] = {}

    def add_node(self, node: LineageNode) -> None:
        self._nodes[node.name] = node

    def get_lineage(self, node_name: str) -> list[str]:
        """Trace backwards from a node to find all ancestors."""
        visited = []
        queue = [node_name]
        while queue:
            current = queue.pop(0)
            if current in visited:
                continue
            visited.append(current)
            node = self._nodes.get(current)
            if node:
                queue.extend(node.inputs)
        return visited

    def to_dict(self) -> dict[str, Any]:
        return {name: asdict(node) for name, node in self._nodes.items()}

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))


def build_reeftwin_lineage() -> DataLineage:
    """Build the standard ReefTwin data lineage graph."""
    lineage = DataLineage()
    lineage.add_node(LineageNode("iot_sensors", "source", outputs=["iot_readings"]))
    lineage.add_node(LineageNode("noaa_crw", "source", outputs=["noaa_data"]))
    lineage.add_node(LineageNode("iot_readings", "transform", inputs=["iot_sensors"], outputs=["daily_aggregates"]))
    lineage.add_node(LineageNode("noaa_data", "transform", inputs=["noaa_crw"], outputs=["heat_stress_features"]))
    lineage.add_node(LineageNode("daily_aggregates", "transform", inputs=["iot_readings"], outputs=["reef_features"]))
    lineage.add_node(LineageNode("heat_stress_features", "transform", inputs=["noaa_data"], outputs=["reef_features"]))
    lineage.add_node(LineageNode("reef_features", "transform",
                                  inputs=["daily_aggregates", "heat_stress_features"],
                                  outputs=["bleaching_model", "hybrid_model"]))
    lineage.add_node(LineageNode("bleaching_model", "model", inputs=["reef_features"], outputs=["predictions"]))
    lineage.add_node(LineageNode("hybrid_model", "model", inputs=["reef_features"], outputs=["predictions"]))
    lineage.add_node(LineageNode("predictions", "output", inputs=["bleaching_model", "hybrid_model"], outputs=["reef_state"]))
    lineage.add_node(LineageNode("reef_state", "output", inputs=["predictions"]))
    return lineage
