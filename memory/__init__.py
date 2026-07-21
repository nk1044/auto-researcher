from __future__ import annotations

import os
from typing import Optional

from shared.types import AgentState, IterationRecord, MemoryEntry, OutcomeType

from .episodic import EpisodicMemory
from .semantic import SemanticMemory
from .state import StateMemory


class Memory:
    """Unified API wrapping all three memory layers."""

    def __init__(
        self,
        data_dir: str,
        ollama_host: str,
        embed_model: str,
        dup_threshold: float = 0.92,
    ) -> None:
        self._dup_threshold = dup_threshold
        os.makedirs(data_dir, exist_ok=True)
        self._episodic = EpisodicMemory(os.path.join(data_dir, "episodic.db"))
        self._semantic = SemanticMemory(
            db_path=os.path.join(data_dir, "semantic_db"),
            ollama_host=ollama_host,
            embed_model=embed_model,
        )
        self._state = StateMemory(os.path.join(data_dir, "state.db"))

    async def init(self) -> None:
        self._episodic.init_db()
        await self._semantic.init()
        self._state.init_db()

    async def record(self, record: IterationRecord) -> None:
        self._episodic.record(record)

        entry = MemoryEntry(
            id=record.id,
            text=record.hypothesis,
            outcome=record.outcome,
            score=record.score,
            remark=record.remark,
            ts=record.ts,
            iteration=record.iteration,
        )
        await self._semantic.store(entry)

    async def retrieve(
        self, query: str, k: int = 5, include_failures: bool = True
    ) -> list[MemoryEntry]:
        return await self._semantic.retrieve(query, k=k, include_failures=include_failures)

    async def is_duplicate_failure(self, hypothesis_text: str) -> bool:
        return await self._semantic.is_duplicate_failure(hypothesis_text, threshold=self._dup_threshold)

    async def top_failures(self, context: str, k: int = 5) -> list[MemoryEntry]:
        return await self._semantic.top_failures(context, k=k)

    async def get_best_wins(self, k: int = 3) -> list[MemoryEntry]:
        return await self._semantic.get_best_wins(k=k)

    async def is_building_on_win(self, hypothesis_text: str, similarity_floor: float = 0.80) -> bool:
        return await self._semantic.is_building_on_win(hypothesis_text, similarity_floor=similarity_floor)

    def load_state(self) -> Optional[AgentState]:
        return self._state.load_state()

    def save_state(self, state: AgentState) -> None:
        self._state.save_state(state)

    def get_recent_iterations(self, k: int = 20) -> list[IterationRecord]:
        return self._episodic.get_recent(k)

    def iteration_count(self) -> int:
        return self._episodic.count()
