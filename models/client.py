from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Optional

import httpx

from shared.types import ModelSpec
from models.registry import ModelRegistry

logger = logging.getLogger(__name__)


@dataclass
class ChatMessage:
    role: str
    content: str
    tool_calls: Optional[list[dict[str, Any]]] = None
    tool_call_id: Optional[str] = None


@dataclass
class ChatResponse:
    content: str
    tool_calls: list[dict[str, Any]]
    model: str
    prompt_tokens: int
    completion_tokens: int


class OllamaClient:
    def __init__(
        self,
        base_url: str,
        registry: ModelRegistry,
        timeout: float = 120.0,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._registry = registry
        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            timeout=httpx.Timeout(timeout),
        )

    async def chat(
        self,
        model_spec: ModelSpec,
        messages: list[ChatMessage],
        tools: Optional[list[dict[str, Any]]] = None,
        options_override: Optional[dict[str, Any]] = None,
    ) -> ChatResponse:
        options: dict[str, Any] = {**model_spec.options}
        if options_override:
            options.update(options_override)

        serialized_messages: list[dict[str, Any]] = []
        for msg in messages:
            entry: dict[str, Any] = {"role": msg.role, "content": msg.content}
            if msg.tool_calls is not None:
                entry["tool_calls"] = msg.tool_calls
            if msg.tool_call_id is not None:
                entry["tool_call_id"] = msg.tool_call_id
            serialized_messages.append(entry)

        body: dict[str, Any] = {
            "model": model_spec.name,
            "messages": serialized_messages,
            "stream": False,
        }
        if tools:
            body["tools"] = tools
        if options:
            body["options"] = options

        logger.debug("OllamaClient.chat model=%s", model_spec.name)
        response = await self._client.post("/api/chat", json=body)
        response.raise_for_status()
        data = response.json()

        message = data.get("message", {})
        content: str = message.get("content", "")
        tool_calls: list[dict[str, Any]] = message.get("tool_calls") or []
        prompt_tokens: int = data.get("prompt_eval_count", 0) or 0
        completion_tokens: int = data.get("eval_count", 0) or 0

        return ChatResponse(
            content=content,
            tool_calls=tool_calls,
            model=data.get("model", model_spec.name),
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )

    async def embed(
        self, text: str, model: Optional[str] = None
    ) -> list[float]:
        target_model = model if model is not None else self._registry.embed_model
        body = {"model": target_model, "prompt": text}
        logger.debug("OllamaClient.embed model=%s", target_model)
        response = await self._client.post("/api/embeddings", json=body)
        response.raise_for_status()
        return response.json()["embedding"]

    @property
    def registry(self) -> ModelRegistry:
        return self._registry

    async def close(self) -> None:
        await self._client.aclose()
