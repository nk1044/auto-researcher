from __future__ import annotations

from shared.types import ModelSpec
from models.registry import ModelRegistry


class ModelRouter:
    """Deterministic skill-based model routing."""

    def __init__(self, registry: ModelRegistry) -> None:
        self._registry = registry

    def select(
        self, required_skills: list[str]
    ) -> tuple[ModelSpec, list[str], bool]:
        workers = self._registry.workers

        if not required_skills or not workers:
            return (self._registry.default, [], True)

        required_set = set(required_skills)
        best_model: ModelSpec | None = None
        best_matched: list[str] = []
        best_score = 0

        for worker in workers:
            matched = [s for s in worker.skills if s in required_set]
            score = len(matched)
            if score > best_score:
                best_score = score
                best_model = worker
                best_matched = matched

        if best_score == 0 or best_model is None:
            return (self._registry.default, [], True)

        return (best_model, best_matched, False)
