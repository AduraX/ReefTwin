"""LLM-powered scenario interpretation.

Translates raw simulation outputs into natural-language summaries
and recommendations for decision-makers. Addresses the JD requirement
for "user interfaces that allow scientists and decision-makers to
interact effectively with digital twins and complex AI outputs."
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from infrastructure.genai.llm import generate
from infrastructure.logging import get_logger

logger = get_logger("genai.scenario_interpreter")


@dataclass
class ScenarioInterpretation:
    summary: str
    risk_assessment: str
    recommendations: list[str]
    model: str
    input_tokens: int
    output_tokens: int


def interpret_simulation(
    simulation_result: dict[str, Any],
    reef_state: dict[str, Any] | None = None,
) -> ScenarioInterpretation:
    """Generate a natural-language interpretation of a simulation result."""

    context_parts = [f"Simulation Result:\n{_format_dict(simulation_result)}"]
    if reef_state:
        context_parts.append(f"\nCurrent Reef State:\n{_format_dict(reef_state)}")

    prompt = (
        f"{''.join(context_parts)}\n\n"
        "Based on this reef simulation, provide:\n"
        "1. A 2-3 sentence plain-language summary of what the scenario means\n"
        "2. A risk assessment (one sentence)\n"
        "3. Three specific management recommendations\n\n"
        "Format your response as:\n"
        "SUMMARY: <summary>\n"
        "RISK: <risk assessment>\n"
        "RECOMMENDATIONS:\n- <recommendation 1>\n- <recommendation 2>\n- <recommendation 3>"
    )

    system = (
        "You are a coral reef management advisor. Translate technical simulation "
        "outputs into actionable advice for reef managers. Be specific and cite "
        "the numbers from the data."
    )

    response = generate(prompt, system=system, max_tokens=512)

    # Parse structured response
    summary, risk, recs = _parse_interpretation(response.content)

    return ScenarioInterpretation(
        summary=summary,
        risk_assessment=risk,
        recommendations=recs,
        model=response.model,
        input_tokens=response.input_tokens,
        output_tokens=response.output_tokens,
    )


def interpret_reef_state(reef_state: dict[str, Any]) -> str:
    """Generate a brief natural-language description of current reef state."""
    prompt = (
        f"Current reef state data:\n{_format_dict(reef_state)}\n\n"
        "Provide a concise 2-sentence status update for a reef manager. "
        "Include the key numbers and what they mean."
    )

    system = "You are a marine science communicator. Be concise and factual."
    response = generate(prompt, system=system, max_tokens=256)
    return response.content


def _format_dict(d: dict[str, Any]) -> str:
    lines = []
    for k, v in d.items():
        if isinstance(v, dict):
            lines.append(f"  {k}:")
            for k2, v2 in v.items():
                lines.append(f"    {k2}: {v2}")
        elif isinstance(v, float):
            lines.append(f"  {k}: {v:.4f}")
        else:
            lines.append(f"  {k}: {v}")
    return "\n".join(lines)


def _parse_interpretation(text: str) -> tuple[str, str, list[str]]:
    """Parse the structured LLM response into components."""
    summary = ""
    risk = ""
    recs = []

    lines = text.strip().split("\n")
    current_section = None

    for line in lines:
        stripped = line.strip()
        if stripped.upper().startswith("SUMMARY:"):
            current_section = "summary"
            summary = stripped[len("SUMMARY:"):].strip()
        elif stripped.upper().startswith("RISK:"):
            current_section = "risk"
            risk = stripped[len("RISK:"):].strip()
        elif stripped.upper().startswith("RECOMMENDATIONS:"):
            current_section = "recs"
        elif current_section == "summary" and stripped and not stripped.startswith("-"):
            summary += " " + stripped
        elif current_section == "risk" and stripped and not stripped.startswith("-"):
            risk += " " + stripped
        elif current_section == "recs" and stripped.startswith("-"):
            recs.append(stripped[1:].strip())

    # Fallback if parsing failed
    if not summary:
        summary = text[:200]
    if not risk:
        risk = "Unable to parse risk assessment."
    if not recs:
        recs = ["Review simulation parameters and consult reef monitoring data."]

    return summary.strip(), risk.strip(), recs
