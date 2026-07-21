from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import httpx

from shared.types import ModelSpec

logger = logging.getLogger(__name__)


@dataclass
class WorkerConfig:
    name: str
    skills: list[str]
    options: dict[str, Any] = field(default_factory=dict)


class ModelRegistry:
    """Loads model configuration from config.yaml."""

    def __init__(self, config: dict[str, Any]) -> None:
        models_cfg: dict[str, Any] = config.get("models", config)

        coordinator_name: str | None = models_cfg.get("coordinator")
        if not coordinator_name:
            raise ValueError("models.coordinator is required in config")

        default_name: str = models_cfg.get("default") or coordinator_name
        if not models_cfg.get("default"):
            logger.info(
                "models.default not set — using coordinator model (%s) for worker fallback",
                coordinator_name,
            )

        self._coordinator = ModelSpec(name=coordinator_name)
        self._default = ModelSpec(name=default_name)
        self._embed_model: str = models_cfg.get("embed", "nomic-embed-text")

        raw_workers: list[dict[str, Any]] = models_cfg.get("workers", [])
        self._workers: list[ModelSpec] = [
            ModelSpec(
                name=w["name"],
                skills=w.get("skills", []),
                options=w.get("options", {}),
            )
            for w in raw_workers
        ]

    async def validate(self, ollama_host: str) -> None:
        """Check that Ollama is reachable and every configured model is available."""
        url = ollama_host.rstrip("/") + "/api/tags"
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(url)
                response.raise_for_status()
                data = response.json()
        except httpx.ConnectError:
            raise RuntimeError(
                f"Cannot connect to Ollama at {ollama_host}. "
                "Make sure Ollama is running (`ollama serve`)."
            )
        except httpx.HTTPStatusError as exc:
            raise RuntimeError(
                f"Ollama returned an unexpected response ({exc.response.status_code}) "
                f"from {url}."
            )

        available: set[str] = {m["name"] for m in data.get("models", [])}

        def _is_available(name: str) -> bool:
            if name in available:
                return True
            # Treat untagged name as equivalent to name:latest
            if ":" not in name:
                return f"{name}:latest" in available
            return False

        required: dict[str, str] = {
            "models.coordinator": self._coordinator.name,
            "models.default": self._default.name,
            "models.embed": self._embed_model,
        }
        for i, w in enumerate(self._workers):
            required[f"models.workers[{i}]"] = w.name

        missing: list[str] = []
        for config_key, name in required.items():
            if not _is_available(name):
                missing.append(f"  {config_key}: '{name}'  →  ollama pull {name}")

        if missing:
            raise ValueError(
                "The following models from config.yaml are not available in Ollama:\n"
                + "\n".join(missing)
                + "\n\nPull the missing models and restart."
            )

    @property
    def coordinator(self) -> ModelSpec:
        return self._coordinator

    @property
    def default(self) -> ModelSpec:
        return self._default

    @property
    def embed_model(self) -> str:
        return self._embed_model

    @property
    def workers(self) -> list[ModelSpec]:
        return self._workers
