"""Reef Decision-Support Agent using ReAct pattern.

A multi-step reasoning agent with tools for reef state queries,
simulation, knowledge search, and bleaching history. The agent
decides which tools to call and in what order to answer complex
reef management questions.

Pattern adapted from llm-twin-enhancements agentic.py.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable

from infrastructure.logging import get_logger

logger = get_logger("genai.agent")

SYSTEM_PROMPT = """\
You are a coral reef decision-support agent. You help marine scientists and \
reef managers analyze reef conditions, assess risks, and plan interventions.

You have access to tools. Use them to gather information before answering.

Guidelines:
1. ALWAYS use search_knowledge_base for science questions before answering.
2. Use query_reef_state to check current conditions of specific reefs.
3. Use run_simulation to project future conditions under scenarios.
4. Use get_stress_breakdown for detailed multi-factor stress analysis.
5. Synthesize information from multiple tools when needed.
6. Be specific — cite data and numbers from tool results.
7. When you have enough information, provide your final answer directly.

Available reef IDs: gbr_heron_reef, gbr_lizard_island, coral_sea_reef
"""


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict[str, Any]
    function: Callable[..., str]

    def to_claude_tool(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": {
                "type": "object",
                "properties": self.parameters,
                "required": list(self.parameters.keys()),
            },
        }


@dataclass
class AgentResult:
    answer: str
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    iterations: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0


def _tool_query_reef_state(reef_id: str) -> str:
    """Query current reef state from the digital twin."""
    from infrastructure.db.factory import get_state_store
    store = get_state_store()
    for state in store.load_states():
        if state["reef_id"] == reef_id:
            return json.dumps(state, indent=2)
    return json.dumps({"error": f"No state found for reef_id={reef_id}"})


def _tool_run_simulation(
    reef_id: str,
    temperature_delta_c: float = 0.0,
    duration_days: int = 21,
    turbidity_delta_pct: float = 0.0,
    ph_delta: float = 0.0,
) -> str:
    """Run a scenario simulation on a reef."""
    from infrastructure.settings import settings
    from infrastructure.db.factory import get_state_store

    store = get_state_store()
    base_state = None
    for state in store.load_states():
        if state["reef_id"] == reef_id:
            base_state = state
            break

    if base_state is None:
        return json.dumps({"error": f"No state found for reef_id={reef_id}"})

    base_risk = float(base_state["bleaching_risk_score"])
    temp_pressure = max(0, temperature_delta_c) * settings.sim_temperature_weight
    duration_pressure = min(duration_days / 90, 1.0) * settings.sim_duration_weight
    turbidity_pressure = max(0, turbidity_delta_pct) / 100 * settings.sim_turbidity_weight
    acid_pressure = max(0, -ph_delta) * settings.sim_acidification_weight

    projected_risk = min(1.0, base_risk + temp_pressure + duration_pressure + turbidity_pressure + acid_pressure)

    result = {
        "reef_id": reef_id,
        "baseline_risk": round(base_risk, 4),
        "projected_risk": round(projected_risk, 4),
        "risk_change": round(projected_risk - base_risk, 4),
        "scenario": {
            "temperature_delta_c": temperature_delta_c,
            "duration_days": duration_days,
            "turbidity_delta_pct": turbidity_delta_pct,
            "ph_delta": ph_delta,
        },
    }
    return json.dumps(result, indent=2)


def _tool_search_knowledge_base(query: str) -> str:
    """Search the reef science knowledge base."""
    from infrastructure.genai.knowledge_base import get_knowledge_base
    kb = get_knowledge_base()
    results = kb.search(query, k=3)
    formatted = []
    for r in results:
        formatted.append(
            f"[{r['metadata'].get('source', '?')} | {r['metadata'].get('topic', '?')}] "
            f"{r['content'][:300]}..."
        )
    return "\n\n".join(formatted) if formatted else "No relevant documents found."


def _tool_get_stress_breakdown(reef_id: str) -> str:
    """Get multi-factor stress analysis for a reef."""
    from infrastructure.db.factory import get_state_store
    from models.stress_scoring import ReefStressModel

    store = get_state_store()
    for state in store.load_states():
        if state["reef_id"] == reef_id:
            model = ReefStressModel()
            breakdown = model.score(state)
            return json.dumps({
                "reef_id": reef_id,
                "total_stress": breakdown.total_score,
                "thermal_stress": breakdown.thermal_score,
                "water_quality_stress": breakdown.water_quality_score,
                "biological_stress": breakdown.biological_score,
                "cumulative_stress": breakdown.cumulative_score,
                "dominant_stressor": breakdown.dominant_stressor,
            }, indent=2)

    return json.dumps({"error": f"No state found for reef_id={reef_id}"})


def build_reef_tools() -> list[Tool]:
    """Create the set of tools available to the reef agent."""
    return [
        Tool(
            name="query_reef_state",
            description="Get the current state of a specific reef including temperature, bleaching risk, and ecosystem status.",
            parameters={
                "reef_id": {"type": "string", "description": "Reef identifier (e.g., gbr_heron_reef)"},
            },
            function=lambda **kw: _tool_query_reef_state(**kw),
        ),
        Tool(
            name="run_simulation",
            description="Simulate a climate scenario on a reef to project future bleaching risk. Specify temperature change, duration, turbidity change, and pH change.",
            parameters={
                "reef_id": {"type": "string", "description": "Reef identifier"},
                "temperature_delta_c": {"type": "number", "description": "Temperature change in °C (positive = warming)"},
                "duration_days": {"type": "integer", "description": "Duration of scenario in days"},
            },
            function=lambda **kw: _tool_run_simulation(**kw),
        ),
        Tool(
            name="search_knowledge_base",
            description="Search the reef science knowledge base for information about coral bleaching, reef monitoring, water quality, climate projections, and reef management.",
            parameters={
                "query": {"type": "string", "description": "Search query about reef science"},
            },
            function=lambda **kw: _tool_search_knowledge_base(**kw),
        ),
        Tool(
            name="get_stress_breakdown",
            description="Get a detailed multi-factor stress analysis for a reef, breaking down thermal, water quality, biological, and cumulative stress components.",
            parameters={
                "reef_id": {"type": "string", "description": "Reef identifier"},
            },
            function=lambda **kw: _tool_get_stress_breakdown(**kw),
        ),
    ]


class ReefAgent:
    """ReAct-style agent for reef decision support.

    Works with any configured LLM provider (Claude, OpenAI, Qwen, Ollama).
    Uses Claude-native tool calling for Claude provider, falls back to
    prompt-based tool calling for other providers.
    """

    def __init__(self, max_iterations: int = 5) -> None:
        self.max_iterations = max_iterations
        self.tools = build_reef_tools()
        self._tool_map = {t.name: t for t in self.tools}

    def run(self, query: str) -> AgentResult:
        """Execute the agent loop for a query."""
        from infrastructure.genai.llm import get_provider, MockProvider, ClaudeProvider

        provider = get_provider()

        if isinstance(provider, MockProvider):
            logger.info("Agent running in mock mode")
            return AgentResult(
                answer=f"[Mock agent response for: {query[:100]}]",
                tool_calls=[],
                iterations=1,
            )

        messages = [{"role": "user", "content": query}]
        claude_tools = [t.to_claude_tool() for t in self.tools]
        tool_calls_log = []
        total_in = 0
        total_out = 0

        for iteration in range(1, self.max_iterations + 1):
            response = provider.generate_with_tools(
                messages=messages,
                tools=claude_tools,
                system=SYSTEM_PROMPT,
            )

            # Normalize response across providers
            if isinstance(provider, ClaudeProvider):
                total_in += response.usage.input_tokens
                total_out += response.usage.output_tokens
                content_blocks, has_tool_use, final_text = self._parse_claude_response(response)
            else:
                # OpenAI-compatible response
                usage = getattr(response, "usage", None)
                if usage:
                    total_in += getattr(usage, "prompt_tokens", 0)
                    total_out += getattr(usage, "completion_tokens", 0)
                content_blocks, has_tool_use, final_text = self._parse_openai_response(response)

            messages.append({"role": "assistant", "content": content_blocks})

            if not has_tool_use:
                return AgentResult(
                    answer=final_text,
                    tool_calls=tool_calls_log,
                    iterations=iteration,
                    total_input_tokens=total_in,
                    total_output_tokens=total_out,
                )

            # Execute tool calls and build tool results
            tool_results = self._execute_tools(content_blocks, tool_calls_log)
            messages.append({"role": "user", "content": tool_results})

        return AgentResult(
            answer="[Agent reached max iterations without final answer]",
            tool_calls=tool_calls_log,
            iterations=self.max_iterations,
            total_input_tokens=total_in,
            total_output_tokens=total_out,
        )

    def _parse_claude_response(self, response) -> tuple[list[dict], bool, str]:
        """Parse Anthropic Claude response into normalized blocks."""
        content_blocks = []
        has_tool_use = False
        text_parts = []

        for block in response.content:
            if block.type == "text":
                content_blocks.append({"type": "text", "text": block.text})
                text_parts.append(block.text)
            elif block.type == "tool_use":
                has_tool_use = True
                content_blocks.append({
                    "type": "tool_use",
                    "id": block.id,
                    "name": block.name,
                    "input": block.input,
                })

        return content_blocks, has_tool_use, " ".join(text_parts)

    def _parse_openai_response(self, response) -> tuple[list[dict], bool, str]:
        """Parse OpenAI-compatible response into Claude-format blocks."""
        import json as _json
        choice = response.choices[0]
        msg = choice.message
        content_blocks = []
        has_tool_use = False
        text = msg.content or ""

        if text:
            content_blocks.append({"type": "text", "text": text})

        if msg.tool_calls:
            has_tool_use = True
            for tc in msg.tool_calls:
                content_blocks.append({
                    "type": "tool_use",
                    "id": tc.id,
                    "name": tc.function.name,
                    "input": _json.loads(tc.function.arguments),
                })

        return content_blocks, has_tool_use, text

    def _execute_tools(self, content_blocks: list[dict], tool_calls_log: list[dict]) -> list[dict]:
        """Execute tool calls from content blocks."""
        tool_results = []
        for block in content_blocks:
            if block.get("type") != "tool_use":
                continue
            tool = self._tool_map.get(block["name"])
            if not tool:
                continue

            logger.info("Agent calling tool: %s(%s)", block["name"], block["input"])
            try:
                result = tool.function(**block["input"])
            except Exception as e:
                result = json.dumps({"error": str(e)})
                logger.error("Tool %s failed: %s", block["name"], e)

            tool_calls_log.append({
                "tool": block["name"],
                "input": block["input"],
                "result_summary": result[:200],
            })
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": block["id"],
                "content": result,
            })

        return tool_results
