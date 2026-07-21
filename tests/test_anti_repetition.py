from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from memory import Memory
from memory.semantic import SemanticMemory
from shared.types import OutcomeType


async def test_is_duplicate_failure_high_similarity_returns_true(tmp_path: Path) -> None:
    mem = Memory(
        data_dir=str(tmp_path / "data"),
        ollama_host="http://localhost:11434",
        embed_model="nomic-embed-text",
    )
    mem._semantic.is_duplicate_failure = AsyncMock(return_value=True)

    result = await mem.is_duplicate_failure("add more dropout regularisation")

    assert result is True
    # Memory forwards the configured dup_threshold (default 0.92) to SemanticMemory
    mem._semantic.is_duplicate_failure.assert_awaited_once_with(
        "add more dropout regularisation", threshold=0.92
    )


async def test_is_duplicate_failure_low_similarity_returns_false(tmp_path: Path) -> None:
    mem = Memory(
        data_dir=str(tmp_path / "data"),
        ollama_host="http://localhost:11434",
        embed_model="nomic-embed-text",
    )
    mem._semantic.is_duplicate_failure = AsyncMock(return_value=False)

    result = await mem.is_duplicate_failure("completely novel hypothesis")

    assert result is False


async def test_is_duplicate_failure_empty_memory_returns_false(tmp_path: Path) -> None:
    mem = Memory(
        data_dir=str(tmp_path / "data"),
        ollama_host="http://localhost:11434",
        embed_model="nomic-embed-text",
    )
    mem._semantic.is_duplicate_failure = AsyncMock(return_value=False)

    result = await mem.is_duplicate_failure("any hypothesis")

    assert result is False


async def test_memory_uses_configured_dup_threshold(tmp_path: Path) -> None:
    mem = Memory(
        data_dir=str(tmp_path / "data"),
        ollama_host="http://localhost:11434",
        embed_model="nomic-embed-text",
        dup_threshold=0.85,
    )
    mem._semantic.is_duplicate_failure = AsyncMock(return_value=True)

    await mem.is_duplicate_failure("some hypothesis")

    mem._semantic.is_duplicate_failure.assert_awaited_once_with(
        "some hypothesis", threshold=0.85
    )


async def test_dup_threshold_above_0_92_returns_true() -> None:
    sem = SemanticMemory(
        db_path="/tmp/_test_sem_above",
        ollama_host="http://localhost:11434",
        embed_model="nomic-embed-text",
    )
    sem.embed = AsyncMock(return_value=[0.0] * 768)

    mock_table = MagicMock()
    mock_table.count_rows.return_value = 3

    mock_chain = MagicMock()
    mock_chain.where.return_value = mock_chain
    mock_chain.limit.return_value = mock_chain
    # distance = 0.05 → cosine_sim = 0.95 > 0.92 → duplicate
    mock_chain.to_list.return_value = [
        {
            "_distance": 0.05,
            "id": "abc",
            "text": "prior hypothesis",
            "outcome": OutcomeType.MISTAKE.value,
            "score": 0.0,
            "remark": "",
            "iteration": 1,
            "ts": "2024-01-01T00:00:00",
        }
    ]
    mock_table.search.return_value = mock_chain
    sem._table = mock_table

    result = await sem.is_duplicate_failure("test hypothesis")

    assert result is True


async def test_dup_threshold_below_0_92_returns_false() -> None:
    sem = SemanticMemory(
        db_path="/tmp/_test_sem_below",
        ollama_host="http://localhost:11434",
        embed_model="nomic-embed-text",
    )
    sem.embed = AsyncMock(return_value=[0.0] * 768)

    mock_table = MagicMock()
    mock_table.count_rows.return_value = 3

    mock_chain = MagicMock()
    mock_chain.where.return_value = mock_chain
    mock_chain.limit.return_value = mock_chain
    # distance = 0.12 → cosine_sim = 0.88 < 0.92 → not duplicate
    mock_chain.to_list.return_value = [
        {
            "_distance": 0.12,
            "id": "def",
            "text": "different hypothesis",
            "outcome": OutcomeType.MISTAKE.value,
            "score": 0.0,
            "remark": "",
            "iteration": 2,
            "ts": "2024-01-02T00:00:00",
        }
    ]
    mock_table.search.return_value = mock_chain
    sem._table = mock_table

    result = await sem.is_duplicate_failure("test hypothesis")

    assert result is False


async def test_dup_threshold_boundary_exactly_0_92_returns_false() -> None:
    sem = SemanticMemory(
        db_path="/tmp/_test_sem_boundary",
        ollama_host="http://localhost:11434",
        embed_model="nomic-embed-text",
    )
    sem.embed = AsyncMock(return_value=[0.0] * 768)

    mock_table = MagicMock()
    mock_table.count_rows.return_value = 1

    mock_chain = MagicMock()
    mock_chain.where.return_value = mock_chain
    mock_chain.limit.return_value = mock_chain
    # distance = 0.08 → cosine_sim = 0.92, threshold is strictly > 0.92 → False
    mock_chain.to_list.return_value = [
        {
            "_distance": 0.08,
            "id": "ghi",
            "text": "boundary hypothesis",
            "outcome": OutcomeType.MISTAKE.value,
            "score": 0.0,
            "remark": "",
            "iteration": 3,
            "ts": "2024-01-03T00:00:00",
        }
    ]
    mock_table.search.return_value = mock_chain
    sem._table = mock_table

    result = await sem.is_duplicate_failure("boundary hypothesis")

    assert result is False


async def test_dup_threshold_empty_table_returns_false() -> None:
    sem = SemanticMemory(
        db_path="/tmp/_test_sem_empty",
        ollama_host="http://localhost:11434",
        embed_model="nomic-embed-text",
    )
    sem.embed = AsyncMock(return_value=[0.0] * 768)

    mock_table = MagicMock()
    mock_table.count_rows.return_value = 0
    sem._table = mock_table

    result = await sem.is_duplicate_failure("anything")

    assert result is False
    mock_table.search.assert_not_called()
