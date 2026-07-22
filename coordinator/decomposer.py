"""Hypothesis decomposition: coordinator calls this to produce SubtaskBriefs."""

from __future__ import annotations

import json
import logging
import re
import uuid
from typing import Any, Optional

from shared.types import ExplorationResult, Hypothesis, SubtaskBrief

logger = logging.getLogger(__name__)

DECOMPOSE_SYSTEM = """You are a software engineering coordinator. Given a hypothesis, decompose it into independent subtasks for parallel execution by subagents.

Rules:
1. Each subtask must be fully self-contained — subagents cannot communicate.
2. No cross-dependencies between subtasks.
3. Use 1 to {max_subagents} subtasks. Use 1 for a focused single-file change.
4. Required fields: goal, scope, constraints, expected_output, required_skills.
5. Skill tags: code, refactor, debug, analysis, performance, math, proof, numeric, planning, docs, testing, security.

CRITICAL — goal must be a precise implementation instruction, not a vague description:
- BAD:  "Improve the SparseMoe class to use more experts"
- GOOD: "In models/moe.py, find the SparseMoe.__init__ method and change num_experts=8 to num_experts=16. Also update any assertion or comment that references the number 8 in the same class."
The goal must name: the exact file, the exact class/function, the exact variable or line to change, and the new value or behaviour. The subagent reads ONLY the files in scope — tell it precisely what to do there.

Output ONLY valid JSON:
{{
  "subtasks": [
    {{
      "goal": "<precise instruction: file, class/function, exact change, why>",
      "scope": ["exact/path/to/file.py"],
      "constraints": "Do not modify test files. Do not change the public API unless the hypothesis requires it.",
      "expected_output": "<what the diff should look like — e.g. 'num_experts changed from 8 to 16 in SparseMoe.__init__'>",
      "required_skills": ["code"]
    }}
  ],
  "split_rationale": "..."
}}
"""

DECOMPOSE_USER = """Hypothesis: {hypothesis}

Target repository: {repo_path}

Files in repo:
{file_listing}

Current file contents — read carefully to find exact class names, function names, and variable values:
{file_contents}

{exploration_section}
{fix_section}
Decompose into {max_subagents_max} or fewer subtasks.
- scope: exact file paths from the listing that the subagent must edit.
- goal: name the exact file, class, function, variable, and the precise change — the subagent gets only these files and nothing else.
- expected_output: describe what the resulting diff should contain.
Remember: subtasks run in parallel, cannot share context, and each subagent sees only its scoped files."""


def _extract_json(text: str) -> dict:
    """Extract first JSON object from model output."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
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
    file_slices: dict[str, str] | None = None,
    exploration: Optional[ExplorationResult] = None,
    fix_attempt: int = 0,
    prev_remark: Optional[str] = None,
) -> list[dict[str, str]]:
    """Build the messages list for the decompose LLM call."""
    if file_slices:
        contents_parts = []
        for path, content in file_slices.items():
            contents_parts.append(f"### {path}\n```\n{content}\n```")
        file_contents = "\n\n".join(contents_parts)
    else:
        file_contents = "(no file contents available)"

    exploration_section = ""
    if exploration:
        exploration_section = (
            f"## Project Architecture (from explorer subagents)\n\n"
            f"### Structure\n{exploration.architecture[:600]}\n\n"
            f"### Test Mechanism\n{exploration.test_structure[:400]}\n\n"
            f"### Key Patterns\n{exploration.key_patterns[:400]}\n\n"
        )

    fix_section = ""
    if fix_attempt > 0 and prev_remark:
        fix_section = (
            f"## Previous Attempt Failed (inner attempt #{fix_attempt})\n\n"
            f"The previous decomposition and implementation did NOT pass the test.\n"
            f"Error/remark from test:\n```\n{prev_remark[:800]}\n```\n\n"
            f"Decompose DIFFERENTLY this time — target different files, fix the specific "
            f"error above, or try a completely different implementation approach.\n\n"
        )

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
                file_listing=file_listing,
                file_contents=file_contents,
                exploration_section=exploration_section,
                fix_section=fix_section,
                max_subagents_max=max_subagents,
            ),
        },
    ]
