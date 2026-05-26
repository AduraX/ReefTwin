"""Reef ecosystem graph model using NetworkX.

Models reef ecosystem relationships as a weighted directed graph:
  - Nodes: reef components (coral, fish, algae, water quality, climate)
  - Edges: ecological interactions with strength weights
  - Stress propagation: simulate how stressors cascade through the ecosystem

Inspired by GPURoute's graph-based topology pattern.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from infrastructure.logging import get_logger

logger = get_logger("models.ecosystem_graph")


@dataclass
class EcosystemNode:
    name: str
    node_type: str  # "organism", "environmental", "stressor"
    health: float = 1.0  # 0.0 = dead, 1.0 = healthy
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class EcosystemEdge:
    source: str
    target: str
    interaction: str  # "symbiosis", "predation", "competition", "stress", "dependency"
    weight: float = 1.0  # strength of interaction


def build_reef_ecosystem_graph():
    """Build a standard coral reef ecosystem interaction graph.

    Returns a NetworkX DiGraph representing ecological relationships.
    """
    import networkx as nx

    G = nx.DiGraph()

    # --- Nodes ---
    nodes = [
        EcosystemNode("coral", "organism", metadata={"description": "Hard coral colonies"}),
        EcosystemNode("zooxanthellae", "organism", metadata={"description": "Symbiotic algae in coral"}),
        EcosystemNode("herbivore_fish", "organism", metadata={"description": "Parrotfish, surgeonfish"}),
        EcosystemNode("predator_fish", "organism", metadata={"description": "Groupers, sharks"}),
        EcosystemNode("macroalgae", "organism", metadata={"description": "Competing algae"}),
        EcosystemNode("cots", "organism", metadata={"description": "Crown-of-thorns starfish"}),
        EcosystemNode("water_temperature", "environmental"),
        EcosystemNode("water_quality", "environmental"),
        EcosystemNode("ocean_acidification", "environmental"),
        EcosystemNode("light_availability", "environmental"),
        EcosystemNode("thermal_stress", "stressor"),
        EcosystemNode("pollution", "stressor"),
        EcosystemNode("bleaching", "stressor"),
    ]
    for node in nodes:
        G.add_node(node.name, **{
            "node_type": node.node_type,
            "health": node.health,
            **node.metadata,
        })

    # --- Edges (ecological interactions) ---
    edges = [
        # Symbiosis
        EcosystemEdge("zooxanthellae", "coral", "symbiosis", 0.9),
        EcosystemEdge("coral", "zooxanthellae", "dependency", 0.8),
        # Food web
        EcosystemEdge("herbivore_fish", "macroalgae", "predation", 0.7),
        EcosystemEdge("predator_fish", "herbivore_fish", "predation", 0.5),
        EcosystemEdge("cots", "coral", "predation", 0.6),
        # Competition
        EcosystemEdge("macroalgae", "coral", "competition", 0.6),
        # Environmental dependencies
        EcosystemEdge("light_availability", "zooxanthellae", "dependency", 0.8),
        EcosystemEdge("water_quality", "coral", "dependency", 0.7),
        EcosystemEdge("water_quality", "light_availability", "dependency", 0.5),
        # Stress pathways
        EcosystemEdge("water_temperature", "thermal_stress", "stress", 0.9),
        EcosystemEdge("thermal_stress", "zooxanthellae", "stress", 0.85),
        EcosystemEdge("thermal_stress", "bleaching", "stress", 0.8),
        EcosystemEdge("bleaching", "coral", "stress", 0.9),
        EcosystemEdge("ocean_acidification", "coral", "stress", 0.6),
        EcosystemEdge("pollution", "water_quality", "stress", 0.7),
        EcosystemEdge("pollution", "macroalgae", "dependency", 0.4),  # nutrients boost algae
    ]
    for edge in edges:
        G.add_edge(edge.source, edge.target, **{
            "interaction": edge.interaction,
            "weight": edge.weight,
        })

    logger.info("Ecosystem graph built: %d nodes, %d edges", G.number_of_nodes(), G.number_of_edges())
    return G


def simulate_stress_propagation(
    G,
    initial_stress: dict[str, float],
    propagation_steps: int = 3,
    decay: float = 0.7,
) -> dict[str, float]:
    """Simulate how stress propagates through the ecosystem graph.

    Args:
        G: NetworkX DiGraph of ecosystem relationships.
        initial_stress: Dict of node_name → initial stress level (0-1).
        propagation_steps: Number of propagation iterations.
        decay: Stress decays by this factor at each hop.

    Returns:
        Dict of node_name → final stress level after propagation.
    """
    stress = {node: 0.0 for node in G.nodes}
    stress.update(initial_stress)

    for step in range(propagation_steps):
        new_stress = dict(stress)
        for source, target, data in G.edges(data=True):
            if stress[source] > 0:
                interaction = data.get("interaction", "")
                weight = data.get("weight", 1.0)
                # Stress propagates through stress/dependency edges
                if interaction in ("stress", "dependency", "competition"):
                    propagated = stress[source] * weight * (decay ** step)
                    new_stress[target] = min(1.0, new_stress[target] + propagated)
        stress = new_stress

    return {k: round(v, 4) for k, v in stress.items() if v > 0}


def get_ecosystem_summary(G) -> dict[str, Any]:
    """Get summary statistics of the ecosystem graph."""

    return {
        "nodes": G.number_of_nodes(),
        "edges": G.number_of_edges(),
        "node_types": {
            ntype: len([n for n, d in G.nodes(data=True) if d.get("node_type") == ntype])
            for ntype in ["organism", "environmental", "stressor"]
        },
        "interaction_types": {
            itype: len([1 for _, _, d in G.edges(data=True) if d.get("interaction") == itype])
            for itype in ["symbiosis", "predation", "competition", "stress", "dependency"]
        },
        "most_connected": sorted(
            [(n, G.degree(n)) for n in G.nodes],
            key=lambda x: -x[1],
        )[:5],
    }
