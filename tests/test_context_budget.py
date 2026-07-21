from __future__ import annotations

import datetime
import random
from pathlib import Path
from datetime import UTC

import pytest

from coordinator.context import CHARS_PER_TOKEN as COORD_CPT
from coordinator.context import assemble_coordinator_context
from shared.types import MemoryEntry, OutcomeType, SubtaskBrief
from subagent.context import CHARS_PER_TOKEN as SUBAGENT_CPT
from subagent.context import assemble_subagent_context

_DEFAULT_COORD_RATIOS: dict[str, float] = {
    "task_spec": 0.20,
    "memory": 0.30,
    "files": 0.35,
    "rolling_summary": 0.15,
}

_DEFAULT_SUBAGENT_RATIOS: dict[str, float] = {
    "task_brief": 0.20,
    "files": 0.50,
    "memory": 0.15,
    "rolling_summary": 0.15,
}


def _make_memory_entry(text: str = "hypothesis", score: float = 0.5) -> MemoryEntry:
    return MemoryEntry(
        text=text,
        outcome=OutcomeType.WIN,
        score=score,
        ts=datetime.datetime.now(UTC),
        iteration=1,
    )


def _make_failure_entry(text: str = "failed hypothesis", score: float = 0.2) -> MemoryEntry:
    return MemoryEntry(
        text=text,
        outcome=OutcomeType.MISTAKE,
        score=score,
        ts=datetime.datetime.now(UTC),
        iteration=1,
    )


def test_coordinator_budget_simple() -> None:
    token_budget = 4096
    result = assemble_coordinator_context(
        task_spec="Improve model accuracy.",
        tool_schemas=[{"name": "run_eval", "description": "run evaluation"}],
        memory_wins=[_make_memory_entry("used dropout regularisation")],
        memory_failures=[_make_failure_entry("increased learning rate — overfit")],
        file_slices={"src/model.py": "class Model:\n    pass\n"},
        rolling_summary="Changed dropout rate from 0.1 to 0.3.",
        token_budget=token_budget,
        ratios=_DEFAULT_COORD_RATIOS,
    )
    assert len(result) <= token_budget * COORD_CPT + 100


def test_coordinator_budget_no_memory() -> None:
    token_budget = 2048
    result = assemble_coordinator_context(
        task_spec="Task with no prior history.",
        tool_schemas=[],
        memory_wins=[],
        memory_failures=[],
        file_slices={},
        rolling_summary="",
        token_budget=token_budget,
        ratios=_DEFAULT_COORD_RATIOS,
    )
    assert len(result) > 0
    assert len(result) <= token_budget * COORD_CPT + 100


def test_coordinator_budget_large_inputs() -> None:
    token_budget = 2000
    long_task = "x" * 50_000
    long_content = "y = 1\n" * 20_000
    big_wins = [_make_memory_entry("w" * 1000) for _ in range(20)]
    big_fails = [_make_failure_entry("f" * 1000) for _ in range(20)]
    result = assemble_coordinator_context(
        task_spec=long_task,
        tool_schemas=[{"name": f"tool_{i}", "params": "a" * 100} for i in range(50)],
        memory_wins=big_wins,
        memory_failures=big_fails,
        file_slices={f"src/module_{i}.py": long_content for i in range(10)},
        rolling_summary="a" * 10_000,
        token_budget=token_budget,
        ratios=_DEFAULT_COORD_RATIOS,
    )
    assert len(result) <= token_budget * COORD_CPT + 100


def test_coordinator_budget_random_100_iterations() -> None:
    rng = random.Random(42)
    for _ in range(100):
        token_budget = rng.randint(512, 16384)
        n_wins = rng.randint(0, 10)
        n_fails = rng.randint(0, 10)
        n_files = rng.randint(0, 8)

        wins = [
            _make_memory_entry("w" * rng.randint(10, 800), score=rng.random())
            for _ in range(n_wins)
        ]
        fails = [
            _make_failure_entry("f" * rng.randint(10, 800), score=rng.random())
            for _ in range(n_fails)
        ]
        file_slices = {
            f"src/file_{i}.py": "line\n" * rng.randint(1, 2000)
            for i in range(n_files)
        }
        task_spec = "t" * rng.randint(50, 5000)
        rolling = "r" * rng.randint(0, 2000)

        result = assemble_coordinator_context(
            task_spec=task_spec,
            tool_schemas=[],
            memory_wins=wins,
            memory_failures=fails,
            file_slices=file_slices,
            rolling_summary=rolling,
            token_budget=token_budget,
            ratios=_DEFAULT_COORD_RATIOS,
        )
        limit = token_budget * COORD_CPT + 100
        assert len(result) <= limit, (
            f"iteration overflow: {len(result)} > {limit} "
            f"(budget={token_budget}, wins={n_wins}, fails={n_fails}, files={n_files})"
        )


def test_coordinator_section_ratio_allocation() -> None:
    token_budget = 4096
    spec_ratio = 0.20
    summary_ratio = 0.15
    result = assemble_coordinator_context(
        task_spec="A" * 100_000,
        tool_schemas=[],
        memory_wins=[],
        memory_failures=[],
        file_slices={},
        rolling_summary="B" * 100_000,
        token_budget=token_budget,
        ratios={
            "task_spec": spec_ratio,
            "memory": 0.30,
            "files": 0.35,
            "rolling_summary": summary_ratio,
        },
    )
    spec_budget = int(token_budget * spec_ratio * COORD_CPT)
    summary_budget = int(token_budget * summary_ratio * COORD_CPT)

    sections = result.split("\n\n")
    spec_section = sections[0]
    summary_section = sections[-1]
    assert len(spec_section) <= spec_budget + 10
    assert len(summary_section) <= summary_budget + 10


def test_subagent_budget_simple() -> None:
    token_budget = 4096
    brief = SubtaskBrief(
        goal="Refactor the data loader.",
        scope=[],
        constraints="No external libraries.",
        expected_output="Cleaner data loader.",
        required_skills=["code"],
    )
    result = assemble_subagent_context(
        brief=brief,
        tool_schemas=[{"name": "read_file", "description": "reads a file"}],
        memory_entries=[_make_memory_entry("previous refactor attempt")],
        rolling_summary="Read 3 files so far.",
        token_budget=token_budget,
        ratios=_DEFAULT_SUBAGENT_RATIOS,
    )
    assert len(result) > 0
    assert len(result) <= token_budget * SUBAGENT_CPT + 100


def test_subagent_empty_scope_produces_valid_output() -> None:
    brief = SubtaskBrief(goal="Do something.", scope=[])
    result = assemble_subagent_context(
        brief=brief,
        tool_schemas=[],
        memory_entries=[],
        rolling_summary="",
        token_budget=2048,
        ratios=_DEFAULT_SUBAGENT_RATIOS,
    )
    assert "Subtask Brief" in result
    assert "In-Scope Files" in result
    assert "no specific scope" in result.lower()


def test_subagent_large_file_truncated_not_dropped(tmp_path: Path) -> None:
    large_file = tmp_path / "big.py"
    large_file.write_text("value = 42\n" * 10_000)

    token_budget = 800
    brief = SubtaskBrief(
        goal="Analyse big.py",
        scope=[str(large_file)],
    )
    result = assemble_subagent_context(
        brief=brief,
        tool_schemas=[],
        memory_entries=[],
        rolling_summary="",
        token_budget=token_budget,
        ratios=_DEFAULT_SUBAGENT_RATIOS,
    )
    assert "big.py" in result
    assert "truncated" in result.lower()
    assert len(result) <= token_budget * SUBAGENT_CPT + 100


def test_subagent_budget_random_100_iterations(tmp_path: Path) -> None:
    rng = random.Random(99)
    for i in range(100):
        token_budget = rng.randint(512, 16384)
        n_entries = rng.randint(0, 8)
        entries = [
            _make_memory_entry("m" * rng.randint(10, 600), score=rng.random())
            for _ in range(n_entries)
        ]
        rolling = "r" * rng.randint(0, 1500)
        brief = SubtaskBrief(
            goal="g" * rng.randint(20, 400),
            scope=[],
            constraints="c" * rng.randint(0, 300),
            expected_output="o" * rng.randint(0, 300),
        )
        result = assemble_subagent_context(
            brief=brief,
            tool_schemas=[],
            memory_entries=entries,
            rolling_summary=rolling,
            token_budget=token_budget,
            ratios=_DEFAULT_SUBAGENT_RATIOS,
        )
        limit = token_budget * SUBAGENT_CPT + 100
        assert len(result) <= limit, (
            f"iter {i} overflow: {len(result)} > {limit} (budget={token_budget})"
        )
