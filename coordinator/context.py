"""Context assembly for the coordinator: bounded, deterministic, no raw history."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from shared.types import MemoryEntry, SubtaskResult

logger = logging.getLogger(__name__)

CHARS_PER_TOKEN = 3


def _budget_chars(token_budget: int, ratio: float) -> int:
    return int(token_budget * ratio * CHARS_PER_TOKEN)


def _truncate(text: str, max_chars: int, label: str) -> str:
    if len(text) <= max_chars:
        return text
    keep = max_chars - 60
    logger.debug("Section %r truncated from %d to %d chars", label, len(text), max_chars)
    return text[:keep] + f"\n... [{label} truncated — {len(text) - keep} chars omitted]"


def read_file_slice(path: str, max_chars: int) -> str:
    try:
        content = Path(path).read_text(encoding="utf-8", errors="replace")
        return content[:max_chars] + (f"\n... [truncated]" if len(content) > max_chars else "")
    except OSError as exc:
        return f"[could not read {path}: {exc}]"


def assemble_coordinator_context(
    task_spec: str,
    tool_schemas: list[dict],
    memory_wins: list[MemoryEntry],
    memory_failures: list[MemoryEntry],
    file_slices: dict[str, str],
    rolling_summary: str,
    token_budget: int,
    ratios: dict[str, float],
) -> str:
    """Assemble the coordinator context for hypothesis formation.

    Section allocations (from ratios, keys: task_spec, memory, files, rolling_summary):
      ~20% task spec + tool schemas
      ~30% retrieved memory (wins + explicitly labeled failures)
      ~35% current baseline file slices + iteration summary
      ~15% rolling summary of current iteration
    """
    spec_chars = _budget_chars(token_budget, ratios.get("task_spec", 0.20))
    memory_chars = _budget_chars(token_budget, ratios.get("memory", 0.30))
    files_chars = _budget_chars(token_budget, ratios.get("files", 0.35))
    summary_chars = _budget_chars(token_budget, ratios.get("rolling_summary", 0.15))

    sections: list[str] = []

    # --- Section 1: Task spec + tool schemas ---
    spec_text = f"## Task Specification\n{task_spec}"
    if tool_schemas:
        spec_text += "\n\n## Available Tools\n" + json.dumps(tool_schemas, indent=2)
    sections.append(_truncate(spec_text, spec_chars, "task_spec"))

    # --- Section 2: Memory (wins and labeled failures) ---
    mem_parts: list[str] = []
    if memory_wins:
        mem_parts.append("## Past Successes (use as inspiration)")
        for e in memory_wins:
            remark = f" — {e.remark}" if e.remark else ""
            mem_parts.append(f"- [WIN score={e.score:.3f}{remark}] {e.text[:400]}")

    if memory_failures:
        mem_parts.append("\n## Past Failures (already tried — do NOT repeat these)")
        for e in memory_failures:
            remark = f" — {e.remark}" if e.remark else ""
            mem_parts.append(
                f"- [ALREADY TRIED, FAILED{remark}] {e.text[:400]}"
            )

    if not mem_parts:
        mem_parts.append("## Memory\n(no prior iterations)")

    sections.append(_truncate("\n".join(mem_parts), memory_chars, "memory"))

    # --- Section 3: File slices ---
    if file_slices:
        per_file = max(200, files_chars // max(len(file_slices), 1))
        file_parts = ["## Current Codebase State (relevant files)"]
        for path, content in file_slices.items():
            if len(content) > per_file:
                content = content[:per_file] + "\n... [truncated]"
            file_parts.append(f"\n### {path}\n```\n{content}\n```")
        files_section = "\n".join(file_parts)
    else:
        files_section = "## Current Codebase State\n(no files loaded)"
    sections.append(_truncate(files_section, files_chars, "files"))

    # --- Section 4: Rolling summary ---
    if rolling_summary:
        summary_section = f"## Current Iteration Summary\n{rolling_summary}"
    else:
        summary_section = "## Current Iteration Summary\n(starting new iteration)"
    sections.append(_truncate(summary_section, summary_chars, "rolling_summary"))

    result = "\n\n".join(sections)
    logger.debug(
        "Coordinator context: %d chars (~%d tokens est), budget %d tokens",
        len(result),
        len(result) // CHARS_PER_TOKEN,
        token_budget,
    )
    return result


def assemble_integration_context(
    hypothesis: str,
    subagent_results: list[SubtaskResult],
    file_slices: dict[str, str],
    token_budget: int,
    ratios: dict[str, float],
) -> str:
    """Assemble context for the coordinator's review+integrate step."""
    files_chars = _budget_chars(token_budget, ratios.get("files", 0.35))
    results_chars = int(token_budget * 0.40 * CHARS_PER_TOKEN)
    spec_chars = int(token_budget * 0.25 * CHARS_PER_TOKEN)

    sections: list[str] = []

    # Hypothesis
    sections.append(
        _truncate(f"## Hypothesis Being Integrated\n{hypothesis}", spec_chars, "hypothesis")
    )

    # Subagent results
    result_parts = ["## Subagent Results (review each — accept, patch, or reject)"]
    for r in subagent_results:
        result_parts.append(
            f"\n### Subtask {r.subtask_id} — status={r.status.value}"
            f"\nSummary: {r.summary}"
            f"\nFiles: {', '.join(r.files_touched) if r.files_touched else 'none'}"
            f"\n```diff\n{r.diff[:1500] if r.diff else '(no changes)'}\n```"
        )
    sections.append(_truncate("\n".join(result_parts), results_chars, "subagent_results"))

    # File slices
    if file_slices:
        per_file = max(200, files_chars // max(len(file_slices), 1))
        file_parts = ["## Current Baseline Files (for conflict resolution)"]
        for path, content in file_slices.items():
            content = content[:per_file] + ("\n... [truncated]" if len(content) > per_file else "")
            file_parts.append(f"\n### {path}\n```\n{content}\n```")
        sections.append(_truncate("\n".join(file_parts), files_chars, "files"))

    return "\n\n".join(sections)
