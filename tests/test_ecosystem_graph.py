"""Tests for reef ecosystem graph model."""

from models.ecosystem_graph import (
    build_reef_ecosystem_graph,
    simulate_stress_propagation,
    get_ecosystem_summary,
)


def test_graph_structure():
    G = build_reef_ecosystem_graph()
    assert G.number_of_nodes() >= 10
    assert G.number_of_edges() >= 10


def test_graph_node_types():
    G = build_reef_ecosystem_graph()
    summary = get_ecosystem_summary(G)
    assert summary["node_types"]["organism"] >= 5
    assert summary["node_types"]["environmental"] >= 3
    assert summary["node_types"]["stressor"] >= 2


def test_graph_interaction_types():
    G = build_reef_ecosystem_graph()
    summary = get_ecosystem_summary(G)
    assert summary["interaction_types"]["symbiosis"] >= 1
    assert summary["interaction_types"]["predation"] >= 2
    assert summary["interaction_types"]["stress"] >= 3


def test_stress_propagation_thermal():
    G = build_reef_ecosystem_graph()
    # Apply thermal stress and see it propagate to coral
    result = simulate_stress_propagation(
        G,
        initial_stress={"water_temperature": 0.8},
        propagation_steps=3,
    )
    assert result.get("thermal_stress", 0) > 0.3
    assert result.get("bleaching", 0) > 0
    assert result.get("coral", 0) > 0  # stress reaches coral


def test_stress_propagation_pollution():
    G = build_reef_ecosystem_graph()
    result = simulate_stress_propagation(
        G,
        initial_stress={"pollution": 0.9},
        propagation_steps=3,
    )
    assert result.get("water_quality", 0) > 0
    assert result.get("macroalgae", 0) > 0  # nutrients boost algae


def test_no_stress_no_propagation():
    G = build_reef_ecosystem_graph()
    result = simulate_stress_propagation(G, initial_stress={}, propagation_steps=3)
    assert all(v == 0 for v in result.values())


def test_most_connected_nodes():
    G = build_reef_ecosystem_graph()
    summary = get_ecosystem_summary(G)
    top = summary["most_connected"]
    assert len(top) >= 3
    # Coral should be one of the most connected nodes
    node_names = [n for n, _ in top]
    assert "coral" in node_names
