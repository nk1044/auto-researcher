"""Hypothesis decomposition: coordinator calls this to produce SubtaskBriefs."""

from __future__ import annotations

import json
import logging
import re
import uuid
from typing import Any

from shared.types import Hypothesis, SubtaskBrief

logger = logging.getLogger(__name__)

DECOMPOSE_SYSTEM = """You are a software engineering coordinator. Given a hypothesis about how to improve a codebase, decompose it into independent subtasks that can be executed in parallel.

Rules:
1. Each subtask must be fully self-contained — subagents cannot communicate with each other.
2. Subtasks must not have cross-dependencies (no "subtask B needs subtask A's output").
3. Use between 1 and {max_subagents} subtasks (use 1 for simple, focused hypotheses).
4. Each subtask must specify: goal, scope (list of file paths), constraints, expected_output, required_skills.
5. Allowed skill tags: math, proof, numeric, code, refactor, debug, reasoning, planning, analysis, docs, testing, security, performance.

Output ONLY valid JSON matching this schema:
{
  "subtasks": [
    {
      "goal": "...",
      "scope": ["path/to/file.py"],
      "constraints": "...",
      "expected_output": "...",
      "required_skills": ["code", "refactor"]
    }
  ],
  "split_rationale": "..."
}
"""

DECOMPOSE_USER = """Hypothesis: {hypothesis}

Target repository: {repo_path}

Relevant files in repo:
{file_listing}

Decompose this hypothesis into {max_subagents_max} or fewer independent subtasks.
Remember: subtasks run in parallel and cannot share context."""


def _extract_json(text: str) -> dict:
    """Extract first JSON object from model output."""
    # Try direct parse first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Find JSON block
    match = re.search(r"\{[\s\S]*\}", text)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    raise ValueError(f"Could not extract JSON from model output: {text[:200]}")


def build_subtask_briefs(
    hypothesis: Hypothesis,
    decomposition: dict,
    max_subagents: int,
) -> list[SubtaskBrief]:
    """Convert raw decomposition dict to typed SubtaskBrief list."""
    subtasks = decomposition.get("subtasks", [])
    if not subtasks:
        raise ValueError("Decomposition returned no subtasks")

    subtasks = subtasks[:max_subagents]
    briefs: list[SubtaskBrief] = []
    for task in subtasks:
        brief = SubtaskBrief(
            id=str(uuid.uuid4()),
            hypothesis_id=hypothesis.id,
            goal=str(task.get("goal", "")),
            scope=list(task.get("scope", [])),
            constraints=str(task.get("constraints", "")),
            expected_output=str(task.get("expected_output", "")),
            required_skills=list(task.get("required_skills", [])),
        )
        briefs.append(brief)
    return briefs


def make_decompose_messages(
    hypothesis: Hypothesis,
    repo_path: str,
    file_listing: str,
    max_subagents: int,
) -> list[dict[str, str]]:
    """Build the messages list for the decompose LLM call."""
    return [
        {
            "role": "system",
            "content": DECOMPOSE_SYSTEM.format(max_subagents=max_subagents),
        },
        {
            "role": "user",
            "content": DECOMPOSE_USER.format(
                hypothesis=hypothesis.text,
                repo_path=repo_path,
                file_listing=file_listing[:2000],
                max_subagents_max=max_subagents,
            ),
        },
    ]
