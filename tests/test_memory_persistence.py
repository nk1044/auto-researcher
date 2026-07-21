from __future__ import annotations

import datetime
from pathlib import Path

import pytest

from memory.episodic import EpisodicMemory
from memory.state import StateMemory
from shared.types import AgentState, IterationRecord, OutcomeType


def _make_record(
    hypothesis: str = "test hypothesis",
    score: float = 0.75,
    outcome: OutcomeType = OutcomeType.WIN,
    iteration: int = 1,
) -> IterationRecord:
    return IterationRecord(
        hypothesis=hypothesis,
        integrated_diff_hash="sha256abc",
        subagent_contribs=[{"subtask_id": "s1", "files": ["src/main.py"]}],
        score=score,
        remark="looks good",
        outcome=outcome,
        baseline_before=0.50,
        ts=datetime.datetime(2024, 6, 1, 12, 0, 0),
        iteration=iteration,
    )


def test_episodic_record_then_get_last(tmp_path: Path) -> None:
    mem = EpisodicMemory(str(tmp_path / "episodic.db"))
    record = _make_record(hypothesis="try larger batch size", score=0.80, iteration=1)

    mem.record(record)

    last = mem.get_last()
    assert last is not None
    assert last.id == record.id
    assert last.hypothesis == "try larger batch size"
    assert last.score == pytest.approx(0.80)
    assert last.outcome == OutcomeType.WIN
    assert last.iteration == 1


def test_episodic_count_increments(tmp_path: Path) -> None:
    mem = EpisodicMemory(str(tmp_path / "episodic.db"))
    assert mem.count() == 0

    mem.record(_make_record(iteration=1))
    assert mem.count() == 1

    mem.record(_make_record(iteration=2))
    assert mem.count() == 2

    mem.record(_make_record(iteration=3))
    assert mem.count() == 3


def test_episodic_get_recent_returns_at_most_k(tmp_path: Path) -> None:
    mem = EpisodicMemory(str(tmp_path / "episodic.db"))
    for i in range(8):
        mem.record(_make_record(iteration=i + 1))

    recent = mem.get_recent(k=3)
    assert len(recent) == 3


def test_episodic_get_recent_returns_latest_first(tmp_path: Path) -> None:
    mem = EpisodicMemory(str(tmp_path / "episodic.db"))
    for i in range(5):
        mem.record(_make_record(iteration=i + 1, score=float(i)))

    recent = mem.get_recent(k=5)
    iterations = [r.iteration for r in recent]
    assert iterations == sorted(iterations, reverse=True)


def test_episodic_get_last_empty_db(tmp_path: Path) -> None:
    mem = EpisodicMemory(str(tmp_path / "episodic.db"))
    assert mem.get_last() is None


def test_episodic_persists_across_instances(tmp_path: Path) -> None:
    db_path = str(tmp_path / "episodic.db")
    record = _make_record(hypothesis="hypothesis X", score=0.88, iteration=7)

    mem1 = EpisodicMemory(db_path)
    mem1.record(record)

    mem2 = EpisodicMemory(db_path)
    assert mem2.count() == 1
    loaded = mem2.get_last()
    assert loaded is not None
    assert loaded.id == record.id
    assert loaded.hypothesis == "hypothesis X"
    assert loaded.score == pytest.approx(0.88)
    assert loaded.iteration == 7


def test_episodic_subagent_contribs_round_trips(tmp_path: Path) -> None:
    mem = EpisodicMemory(str(tmp_path / "episodic.db"))
    contribs = [
        {"subtask_id": "s1", "files": ["src/a.py", "src/b.py"]},
        {"subtask_id": "s2", "files": []},
    ]
    record = IterationRecord(
        hypothesis="multi-subtask hypothesis",
        integrated_diff_hash="deadbeef",
        subagent_contribs=contribs,
        score=0.60,
        outcome=OutcomeType.NEUTRAL,
        baseline_before=0.55,
        iteration=2,
    )
    mem.record(record)

    loaded = mem.get_last()
    assert loaded is not None
    assert loaded.subagent_contribs == contribs


def test_state_save_then_load(tmp_path: Path) -> None:
    state_mem = StateMemory(str(tmp_path / "state.db"))
    state = AgentState(
        iteration=5,
        baseline_score=0.72,
        working_commit="abc123def456",
        run_id="run-007",
    )
    state_mem.save_state(state)

    loaded = state_mem.load_state()
    assert loaded is not None
    assert loaded.iteration == 5
    assert loaded.baseline_score == pytest.approx(0.72)
    assert loaded.working_commit == "abc123def456"
    assert loaded.run_id == "run-007"


def test_state_load_on_empty_db_returns_none(tmp_path: Path) -> None:
    state_mem = StateMemory(str(tmp_path / "state.db"))
    assert state_mem.load_state() is None


def test_state_save_overwrites_previous(tmp_path: Path) -> None:
    state_mem = StateMemory(str(tmp_path / "state.db"))
    state_mem.save_state(AgentState(iteration=1, baseline_score=0.5))
    state_mem.save_state(AgentState(iteration=2, baseline_score=0.6))

    loaded = state_mem.load_state()
    assert loaded is not None
    assert loaded.iteration == 2
    assert loaded.baseline_score == pytest.approx(0.6)


def test_state_persists_across_instances(tmp_path: Path) -> None:
    db_path = str(tmp_path / "state.db")
    state = AgentState(
        iteration=9,
        baseline_score=0.91,
        working_commit="feedcafe",
        run_id="run-persist",
    )

    StateMemory(db_path).save_state(state)

    loaded = StateMemory(db_path).load_state()
    assert loaded is not None
    assert loaded.iteration == 9
    assert loaded.run_id == "run-persist"
    assert loaded.working_commit == "feedcafe"
