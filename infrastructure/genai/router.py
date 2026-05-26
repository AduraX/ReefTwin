"""Query complexity router for multi-model selection.

Routes reef queries to appropriate processing paths based on
complexity. Simple lookups go directly to the API, moderate
questions use RAG, and complex multi-step queries use the agent.

Pattern adapted from llm-twin-enhancements model_router.py.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Any

from infrastructure.logging import get_logger

logger = get_logger("genai.router")


class QueryComplexity(Enum):
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"


@dataclass
class RoutingDecision:
    complexity: QueryComplexity
    handler: str  # "api_lookup", "rag", or "agent"
    reasoning: str


# Keywords/patterns that indicate complexity
_SIMPLE_PATTERNS = [
    r"^(what is|what's) the (current )?(state|status|temperature|risk)",
    r"^(get|show|list) (reef|reefs|state)",
    r"^(check|look up) .{0,30}$",
]

_COMPLEX_PATTERNS = [
    r"(compare|analyze|assess).*(multiple|several|all) reefs",
    r"(what if|simulate|scenario|project).*and.*(what if|simulate|then)",
    r"(plan|strategy|recommend|advise).*(intervention|management|restoration)",
    r"(why|explain|how does).*(cause|relationship|interact|affect).*and",
]


class QueryRouter:
    """Classifies query complexity and routes to the appropriate handler."""

    def route(self, query: str) -> RoutingDecision:
        query_lower = query.lower().strip()
        query_len = len(query_lower.split())

        # Check simple patterns
        for pattern in _SIMPLE_PATTERNS:
            if re.search(pattern, query_lower):
                return RoutingDecision(
                    complexity=QueryComplexity.SIMPLE,
                    handler="api_lookup",
                    reasoning="Query matches simple lookup pattern",
                )

        # Check complex patterns
        for pattern in _COMPLEX_PATTERNS:
            if re.search(pattern, query_lower):
                return RoutingDecision(
                    complexity=QueryComplexity.COMPLEX,
                    handler="agent",
                    reasoning="Query requires multi-step reasoning or tool use",
                )

        # Heuristic: very short queries are simple, very long are complex
        if query_len <= 6:
            return RoutingDecision(
                complexity=QueryComplexity.SIMPLE,
                handler="api_lookup",
                reasoning="Short query — likely a simple lookup",
            )

        if query_len >= 25 or "?" in query_lower and query_lower.count("?") > 1:
            return RoutingDecision(
                complexity=QueryComplexity.COMPLEX,
                handler="agent",
                reasoning="Long or multi-part query — using agent for multi-step reasoning",
            )

        # Default: moderate → RAG
        return RoutingDecision(
            complexity=QueryComplexity.MODERATE,
            handler="rag",
            reasoning="Standard question — using RAG retrieval",
        )


def route_and_execute(query: str) -> dict[str, Any]:
    """Route a query and execute it through the appropriate handler."""
    router = QueryRouter()
    decision = router.route(query)

    logger.info(
        "Routed query to %s (complexity=%s): %s",
        decision.handler, decision.complexity.value, decision.reasoning,
    )

    result: dict[str, Any] = {
        "query": query,
        "routing": {
            "complexity": decision.complexity.value,
            "handler": decision.handler,
            "reasoning": decision.reasoning,
        },
    }

    if decision.handler == "api_lookup":
        # Direct state lookup — extract reef_id from query
        from infrastructure.db.factory import get_state_store
        store = get_state_store()
        states = store.load_states()
        # Try to find a reef_id mentioned in the query
        for state in states:
            if state["reef_id"] in query.lower().replace(" ", "_"):
                result["answer"] = state
                return result
        result["answer"] = {"reefs": states}
        return result

    elif decision.handler == "rag":
        from infrastructure.genai.rag import HybridRAGPipeline
        pipeline = HybridRAGPipeline()
        rag_result = pipeline.query(query)
        result["answer"] = rag_result.answer
        result["sources"] = rag_result.sources
        result["model"] = rag_result.model
        return result

    elif decision.handler == "agent":
        from infrastructure.genai.agent import ReefAgent
        agent = ReefAgent()
        agent_result = agent.run(query)
        result["answer"] = agent_result.answer
        result["tool_calls"] = agent_result.tool_calls
        result["iterations"] = agent_result.iterations
        return result

    return result
