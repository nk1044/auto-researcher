"""Shared data types used across all modules."""

from __future__ import annotations

import datetime
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


def _new_id() -> str:
    return str(uuid.uuid4())


class TaskStatus(str, Enum):
    SUCCESS = "success"
    FAILED = "failed"
    PARTIAL = "partial"
    SKIPPED = "skipped"


class OutcomeType(str, Enum):
    WIN = "win"
    MISTAKE = "mistake"
    NEUTRAL = "neutral"


@dataclass
class ModelSpec:
    name: str
    skills: list[str] = field(default_factory=list)
    options: dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        return f"ModelSpec({self.name!r}, skills={self.skills})"


@dataclass
class SubtaskBrief:
    """Assigned to a subagent by the coordinator."""

    id: str = field(default_factory=_new_id)
    hypothesis_id: str = ""
    goal: str = ""
    scope: list[str] = field(default_factory=list)  # file/module paths
    constraints: str = ""
    expected_output: str = ""
    required_skills: list[str] = field(default_factory=list)
    model: Optional[ModelSpec] = None  # set by router
    matched_skills: list[str] = field(default_factory=list)  # skills that triggered routing
    fallback: bool = False  # True if default model was used


@dataclass
class SubtaskResult:
    """Returned by a subagent to the coordinator."""

    subtask_id: str
    diff: str = ""
    files_touched: list[str] = field(default_factory=list)
    summary: str = ""
    status: TaskStatus = TaskStatus.FAILED
    error: Optional[str] = None
    steps_taken: int = 0


@dataclass
class Hypothesis:
    id: str = field(default_factory=_new_id)
    text: str = ""
    rationale: str = ""
    iteration: int = 0
    embedding: Optional[list[float]] = None


@dataclass
class IntegrationResult:
    hypothesis_id: str
    diff: str = ""
    files_touched: list[str] = field(default_factory=list)
    summary: str = ""
    path: str = ""  # path to integration worktree
    accepted_subtasks: list[str] = field(default_factory=list)
    rejected_subtasks: list[str] = field(default_factory=list)


@dataclass
class MemoryEntry:
    id: str = field(default_factory=_new_id)
    text: str = ""
    outcome: OutcomeType = OutcomeType.NEUTRAL
    score: float = 0.0
    remark: Optional[str] = None
    embedding: Optional[list[float]] = None
    ts: datetime.datetime = field(default_factory=datetime.datetime.utcnow)
    iteration: int = 0


@dataclass
class AgentState:
    iteration: int = 0
    baseline_score: float = 0.0
    working_commit: str = ""
    run_id: str = field(default_factory=_new_id)


@dataclass
class IterationRecord:
    id: str = field(default_factory=_new_id)
    hypothesis: str = ""
    integrated_diff_hash: str = ""
    subagent_contribs: list[dict[str, Any]] = field(default_factory=list)
    score: float = 0.0
    remark: Optional[str] = None
    outcome: OutcomeType = OutcomeType.NEUTRAL
    baseline_before: float = 0.0
    ts: datetime.datetime = field(default_factory=datetime.datetime.utcnow)
    iteration: int = 0
