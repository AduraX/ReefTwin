"""Pluggable LLM provider for ReefTwin GenAI features.

Providers:
    - claude:  Anthropic Claude API (default)
    - openai:  OpenAI / Codex / GPT-4o / o3
    - qwen:    Alibaba Qwen (via OpenAI-compatible DashScope endpoint)
    - ollama:  Local models via Ollama (Qwen, Llama, Mistral, etc.)
    - mock:    No-op mock for testing (no API key needed)

Selection via REEFTWIN_LLM_PROVIDER + REEFTWIN_LLM_MODEL env vars.
All providers implement the LLMProvider abstract interface.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from infrastructure.logging import get_logger

logger = get_logger("genai.llm")


@dataclass
class LLMResponse:
    content: str
    model: str
    input_tokens: int
    output_tokens: int
    provider: str = ""


class LLMProvider(ABC):
    """Abstract LLM provider interface."""

    @property
    @abstractmethod
    def provider_name(self) -> str: ...

    @abstractmethod
    def generate(
        self,
        prompt: str,
        system: str = "",
        model: str = "",
        max_tokens: int = 1024,
        temperature: float = 0.3,
    ) -> LLMResponse: ...

    @abstractmethod
    def generate_with_tools(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        system: str = "",
        model: str = "",
        max_tokens: int = 1024,
    ) -> Any: ...


# ---------------------------------------------------------------------------
# Mock Provider (testing / offline)
# ---------------------------------------------------------------------------

class MockProvider(LLMProvider):
    @property
    def provider_name(self) -> str:
        return "mock"

    def generate(self, prompt: str, system: str = "", model: str = "",
                 max_tokens: int = 1024, temperature: float = 0.3) -> LLMResponse:
        return LLMResponse(
            content=f"[Mock LLM response for: {prompt[:100]}...]",
            model="mock",
            input_tokens=0,
            output_tokens=0,
            provider="mock",
        )

    def generate_with_tools(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]],
                            system: str = "", model: str = "", max_tokens: int = 1024) -> Any:
        return {
            "content": [{"type": "text", "text": "[Mock tool response]"}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 0, "output_tokens": 0},
        }


# ---------------------------------------------------------------------------
# Claude Provider (Anthropic)
# ---------------------------------------------------------------------------

class ClaudeProvider(LLMProvider):
    """Anthropic Claude API provider.

    Models: claude-sonnet-4-20250514, claude-opus-4-20250514,
            claude-haiku-4-5-20251001, etc.
    """

    def __init__(self) -> None:
        import anthropic
        api_key = os.getenv("ANTHROPIC_API_KEY", "")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY not set")
        self._client = anthropic.Anthropic(api_key=api_key)

    @property
    def provider_name(self) -> str:
        return "claude"

    def generate(self, prompt: str, system: str = "", model: str = "",
                 max_tokens: int = 1024, temperature: float = 0.3) -> LLMResponse:
        from infrastructure.settings import settings
        model = model or settings.llm_model
        kwargs: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system:
            kwargs["system"] = system

        response = self._client.messages.create(**kwargs)
        return LLMResponse(
            content=response.content[0].text,
            model=response.model,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            provider="claude",
        )

    def generate_with_tools(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]],
                            system: str = "", model: str = "", max_tokens: int = 1024) -> Any:
        from infrastructure.settings import settings
        model = model or settings.llm_model
        kwargs: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": messages,
            "tools": tools,
        }
        if system:
            kwargs["system"] = system
        return self._client.messages.create(**kwargs)


# ---------------------------------------------------------------------------
# OpenAI-Compatible Provider (OpenAI, Codex, Azure OpenAI)
# ---------------------------------------------------------------------------

class OpenAIProvider(LLMProvider):
    """OpenAI / Codex / GPT-4o / o3 provider.

    Models: gpt-4o, gpt-4o-mini, o3, o3-mini, codex-mini-latest, etc.
    Also works with Azure OpenAI by setting OPENAI_BASE_URL.
    """

    def __init__(self, api_key: str = "", base_url: str = "") -> None:
        from infrastructure.settings import settings
        key = api_key or settings.openai_api_key or os.getenv("OPENAI_API_KEY", "")
        url = base_url or settings.openai_base_url
        if not key:
            raise ValueError("OPENAI_API_KEY not set")
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("Install openai: pip install openai")
        self._client = OpenAI(api_key=key, base_url=url)

    @property
    def provider_name(self) -> str:
        return "openai"

    def generate(self, prompt: str, system: str = "", model: str = "",
                 max_tokens: int = 1024, temperature: float = 0.3) -> LLMResponse:
        from infrastructure.settings import settings
        model = model or settings.llm_model
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        response = self._client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        choice = response.choices[0]
        usage = response.usage
        return LLMResponse(
            content=choice.message.content or "",
            model=response.model,
            input_tokens=usage.prompt_tokens if usage else 0,
            output_tokens=usage.completion_tokens if usage else 0,
            provider="openai",
        )

    def generate_with_tools(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]],
                            system: str = "", model: str = "", max_tokens: int = 1024) -> Any:
        from infrastructure.settings import settings
        model = model or settings.llm_model

        # Convert Claude tool format to OpenAI function format
        oai_tools = []
        for tool in tools:
            oai_tools.append({
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool.get("description", ""),
                    "parameters": tool.get("input_schema", {}),
                },
            })

        # Convert Claude message format to OpenAI format
        oai_messages = []
        if system:
            oai_messages.append({"role": "system", "content": system})
        for msg in messages:
            if msg["role"] == "user":
                if isinstance(msg["content"], str):
                    oai_messages.append(msg)
                elif isinstance(msg["content"], list):
                    # Tool results
                    for block in msg["content"]:
                        if block.get("type") == "tool_result":
                            oai_messages.append({
                                "role": "tool",
                                "tool_call_id": block.get("tool_use_id", ""),
                                "content": block.get("content", ""),
                            })
            elif msg["role"] == "assistant":
                if isinstance(msg["content"], list):
                    text_parts = []
                    tool_calls = []
                    for block in msg["content"]:
                        if block.get("type") == "text":
                            text_parts.append(block["text"])
                        elif block.get("type") == "tool_use":
                            import json
                            tool_calls.append({
                                "id": block["id"],
                                "type": "function",
                                "function": {
                                    "name": block["name"],
                                    "arguments": json.dumps(block["input"]),
                                },
                            })
                    oai_msg: dict[str, Any] = {"role": "assistant", "content": " ".join(text_parts) or None}
                    if tool_calls:
                        oai_msg["tool_calls"] = tool_calls
                    oai_messages.append(oai_msg)

        response = self._client.chat.completions.create(
            model=model,
            messages=oai_messages,
            tools=oai_tools if oai_tools else None,
            max_tokens=max_tokens,
        )
        return response


# ---------------------------------------------------------------------------
# Qwen Provider (Alibaba DashScope — OpenAI-compatible)
# ---------------------------------------------------------------------------

class QwenProvider(OpenAIProvider):
    """Alibaba Qwen via DashScope OpenAI-compatible endpoint.

    Models: qwen-plus, qwen-turbo, qwen-max, qwen3-235b-a22b, etc.
    Also works with local Qwen via Ollama by pointing QWEN_BASE_URL to Ollama.
    """

    def __init__(self) -> None:
        from infrastructure.settings import settings
        key = settings.qwen_api_key or os.getenv("QWEN_API_KEY", "")
        url = settings.qwen_base_url
        if not key:
            raise ValueError("QWEN_API_KEY not set")
        super().__init__(api_key=key, base_url=url)

    @property
    def provider_name(self) -> str:
        return "qwen"


# ---------------------------------------------------------------------------
# Ollama Provider (local models)
# ---------------------------------------------------------------------------

class OllamaProvider(LLMProvider):
    """Ollama local model provider.

    Models: qwen3:8b, llama3.1:8b, mistral:7b, gemma2:9b, etc.
    Runs locally — no API key needed.
    """

    def __init__(self) -> None:
        from infrastructure.settings import settings
        self._base_url = settings.ollama_base_url
        logger.info("Ollama provider: %s", self._base_url)

    @property
    def provider_name(self) -> str:
        return "ollama"

    def generate(self, prompt: str, system: str = "", model: str = "",
                 max_tokens: int = 1024, temperature: float = 0.3) -> LLMResponse:
        import json
        import urllib.request
        from infrastructure.settings import settings
        model = model or settings.llm_model

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        payload = json.dumps({
            "model": model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": temperature, "num_predict": max_tokens},
        }).encode()

        req = urllib.request.Request(
            f"{self._base_url}/api/chat",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req) as resp:
            data = json.loads(resp.read())

        return LLMResponse(
            content=data.get("message", {}).get("content", ""),
            model=data.get("model", model),
            input_tokens=data.get("prompt_eval_count", 0),
            output_tokens=data.get("eval_count", 0),
            provider="ollama",
        )

    def generate_with_tools(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]],
                            system: str = "", model: str = "", max_tokens: int = 1024) -> Any:
        # Ollama supports tools via /api/chat — same format as OpenAI
        import json
        import urllib.request
        from infrastructure.settings import settings
        model = model or settings.llm_model

        chat_messages = []
        if system:
            chat_messages.append({"role": "system", "content": system})
        for msg in messages:
            if isinstance(msg.get("content"), str):
                chat_messages.append(msg)
            elif isinstance(msg.get("content"), list):
                # Flatten tool results into text
                parts = []
                for block in msg["content"]:
                    if isinstance(block, dict):
                        parts.append(block.get("content", block.get("text", str(block))))
                    else:
                        parts.append(str(block))
                chat_messages.append({"role": msg["role"], "content": "\n".join(parts)})

        # Convert tools to Ollama format
        ollama_tools = []
        for tool in tools:
            ollama_tools.append({
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool.get("description", ""),
                    "parameters": tool.get("input_schema", {}),
                },
            })

        payload = json.dumps({
            "model": model,
            "messages": chat_messages,
            "tools": ollama_tools,
            "stream": False,
        }).encode()

        req = urllib.request.Request(
            f"{self._base_url}/api/chat",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req) as resp:
            return json.loads(resp.read())


# ---------------------------------------------------------------------------
# Qwen Local Provider (HuggingFace transformers — no Ollama/API needed)
# ---------------------------------------------------------------------------

class QwenLocalProvider(LLMProvider):
    """Run Qwen locally via HuggingFace transformers.

    Models: Qwen/Qwen3-0.6B, Qwen/Qwen3-1.7B, Qwen/Qwen3-4B, Qwen/Qwen3-8B, etc.
    Downloads the model on first use (~0.6-16GB depending on size).
    Requires: pip install transformers torch

    Usage:
        REEFTWIN_LLM_PROVIDER=qwen-local
        REEFTWIN_LLM_MODEL=Qwen/Qwen3-0.6B   # smallest, runs on any CPU
    """

    def __init__(self, model_name: str = "") -> None:
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError:
            raise ImportError("Install transformers: pip install transformers torch")

        from infrastructure.settings import settings
        self._model_name = model_name or settings.llm_model or "Qwen/Qwen3-0.6B"

        logger.info("Loading Qwen local model: %s (this may download on first run)...", self._model_name)
        self._tokenizer = AutoTokenizer.from_pretrained(self._model_name)
        self._model = AutoModelForCausalLM.from_pretrained(
            self._model_name,
            torch_dtype="auto",
            device_map="auto",
        )
        logger.info("Qwen local model loaded: %s", self._model_name)

    @property
    def provider_name(self) -> str:
        return "qwen-local"

    def generate(self, prompt: str, system: str = "", model: str = "",
                 max_tokens: int = 1024, temperature: float = 0.3) -> LLMResponse:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        text = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        inputs = self._tokenizer(text, return_tensors="pt").to(self._model.device)
        input_len = inputs["input_ids"].shape[1]

        import torch
        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=max(temperature, 0.01),
                do_sample=temperature > 0,
                top_p=0.9,
            )

        generated = outputs[0][input_len:]
        content = self._tokenizer.decode(generated, skip_special_tokens=True)

        # Qwen3 may include <think>...</think> blocks — strip them
        import re
        content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()

        return LLMResponse(
            content=content,
            model=self._model_name,
            input_tokens=input_len,
            output_tokens=len(generated),
            provider="qwen-local",
        )

    def generate_with_tools(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]],
                            system: str = "", model: str = "", max_tokens: int = 1024) -> Any:
        # For tool use, flatten to a single prompt and parse JSON from response
        tool_desc = "\n".join(
            f"- {t['name']}: {t.get('description', '')} (params: {t.get('input_schema', {})})"
            for t in tools
        )
        combined_system = (
            f"{system}\n\nYou have these tools:\n{tool_desc}\n\n"
            "To use a tool, respond with JSON: {\"tool\": \"name\", \"arguments\": {...}}\n"
            "When you have a final answer, just respond normally without JSON."
        )

        # Extract user content from messages
        user_text = ""
        for msg in messages:
            if msg["role"] == "user":
                if isinstance(msg["content"], str):
                    user_text = msg["content"]
                elif isinstance(msg["content"], list):
                    parts = [b.get("content", "") for b in msg["content"] if isinstance(b, dict)]
                    user_text = "\n".join(parts)

        response = self.generate(user_text, system=combined_system, max_tokens=max_tokens)

        # Try to parse tool call from response
        import json
        content = response.content
        try:
            parsed = json.loads(content)
            if "tool" in parsed:
                return type("FakeResponse", (), {
                    "content": [type("Block", (), {
                        "type": "tool_use",
                        "id": "local-0",
                        "name": parsed["tool"],
                        "input": parsed.get("arguments", {}),
                    })()],
                    "stop_reason": "tool_use",
                    "usage": type("Usage", (), {
                        "input_tokens": response.input_tokens,
                        "output_tokens": response.output_tokens,
                    })(),
                })()
        except (json.JSONDecodeError, KeyError):
            pass

        # Return as text response (final answer)
        return type("FakeResponse", (), {
            "content": [type("Block", (), {"type": "text", "text": content})()],
            "stop_reason": "end_turn",
            "usage": type("Usage", (), {
                "input_tokens": response.input_tokens,
                "output_tokens": response.output_tokens,
            })(),
        })()


# ---------------------------------------------------------------------------
# Factory + Module-level API
# ---------------------------------------------------------------------------

_provider: LLMProvider | None = None


def get_provider(provider_name: str | None = None) -> LLMProvider:
    """Factory for creating the configured LLM provider."""
    global _provider
    if _provider is not None and provider_name is None:
        return _provider

    from infrastructure.settings import settings
    name = provider_name or settings.llm_provider

    try:
        if name == "claude":
            p = ClaudeProvider()
        elif name == "openai":
            p = OpenAIProvider()
        elif name == "qwen":
            p = QwenProvider()
        elif name == "ollama":
            p = OllamaProvider()
        elif name == "qwen-local":
            p = QwenLocalProvider()
        elif name == "mock":
            p = MockProvider()
        else:
            raise ValueError(
                f"Unknown LLM provider: {name!r}. "
                f"Options: claude, openai, qwen, qwen-local, ollama, mock"
            )
    except (ValueError, ImportError) as e:
        logger.warning("Failed to init %s provider (%s) — falling back to mock", name, e)
        p = MockProvider()

    if provider_name is None:
        _provider = p

    logger.info("LLM provider: %s (model: %s)", p.provider_name, settings.llm_model)
    return p


def generate(
    prompt: str,
    system: str = "",
    model: str = "",
    max_tokens: int = 1024,
    temperature: float = 0.3,
) -> LLMResponse:
    """Generate text using the configured LLM provider.

    Provider and model selected via env vars:
        REEFTWIN_LLM_PROVIDER=claude|openai|qwen|ollama|mock
        REEFTWIN_LLM_MODEL=claude-sonnet-4-20250514|gpt-4o|qwen-plus|qwen3:8b|...
    """
    provider = get_provider()
    return provider.generate(prompt, system=system, model=model,
                             max_tokens=max_tokens, temperature=temperature)
