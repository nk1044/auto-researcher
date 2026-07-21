"""Context assembly for subagents: bounded, deterministic, never accumulating."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from shared.types import MemoryEntry, SubtaskBrief

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# Characters-per-token approximation (conservative; 1 token ≈ 3.5 chars for code)
CHARS_PER_TOKEN = 3


def _budget_chars(token_budget: int, ratio: float) -> int:
    return int(token_budget * ratio * CHARS_PER_TOKEN)


def _truncate(text: str, max_chars: int, label: str) -> str:
    if len(text) <= max_chars:
        return text
    keep = max_chars - 40
    truncated = text[:keep]
    logger.debug("Section %r truncated from %d to %d chars", label, len(text), max_chars)
    return truncated + f"\n... [truncated — {len(text) - keep} chars omitted]"


def read_file_slice(path: str, max_chars: int) -> str:
    try:
        content = Path(path).read_text(encoding="utf-8", errors="replace")
        if len(content) > max_chars:
            content = content[:max_chars] + f"\n... [truncated at {max_chars} chars]"
        return content
    except OSError as exc:
        return f"[could not read {path}: {exc}]"


def assemble_subagent_context(
    brief: SubtaskBrief,
    tool_schemas: list[dict],
    memory_entries: list[MemoryEntry],
    rolling_summary: str,
    token_budget: int,
    ratios: dict[str, float],
) -> str:
    """Assemble a subagent's full context string under token_budget.

    Section allocations (from ratios dict, keys: task_brief, files, memory, rolling_summary):
      ~20% task brief + tool schemas
      ~50% in-scope file slices (fresh from disk)
      ~15% narrowly-relevant retrieved memory
      ~15% rolling self-summary

    Never includes: sibling subagent logs, global transcript, coordinator internals.
    """
    task_brief_chars = _budget_chars(token_budget, ratios.get("task_brief", 0.20))
    files_chars = _budget_chars(token_budget, ratios.get("files", 0.50))
    memory_chars = _budget_chars(token_budget, ratios.get("memory", 0.15))
    summary_chars = _budget_chars(token_budget, ratios.get("rolling_summary", 0.15))

    sections: list[str] = []

    # --- Section 1: Task brief + tool schemas ---
    import json

    brief_text = (
        f"## Subtask Brief\n"
        f"ID: {brief.id}\n"
        f"Goal: {brief.goal}\n"
        f"Scope (files/modules): {', '.join(brief.scope) if brief.scope else 'not specified'}\n"
        f"Constraints: {brief.constraints}\n"
        f"Expected output: {brief.expected_output}\n"
        f"Required skills: {', '.join(brief.required_skills) if brief.required_skills else 'none'}\n"
    )
    if tool_schemas:
        tools_text = "\n## Available Tools\n" + json.dumps(tool_schemas, indent=2)
        combined = brief_text + tools_text
    else:
        combined = brief_text
    sections.append(_truncate(combined, task_brief_chars, "task_brief"))

    # --- Section 2: In-scope file slices ---
    if brief.scope:
        per_file_chars = max(200, files_chars // max(len(brief.scope), 1))
        file_parts: list[str] = ["## In-Scope Files (current state)"]
        for path in brief.scope:
            content = read_file_slice(path, per_file_chars)
            file_parts.append(f"\n### {path}\n```\n{content}\n```")
        file_section = "\n".join(file_parts)
    else:
        file_section = "## In-Scope Files\n(no specific scope assigned)"
    sections.append(_truncate(file_section, files_chars, "files"))

    # --- Section 3: Retrieved memory ---
    if memory_entries:
        mem_parts = ["## Relevant Past Iterations"]
        for entry in memory_entries:
            outcome_label = f"[{entry.outcome.upper()}]" if hasattr(entry.outcome, "upper") else f"[{entry.outcome}]"
            remark = f" — {entry.remark}" if entry.remark else ""
            mem_parts.append(
                f"- {outcome_label} score={entry.score:.3f}{remark}: {entry.text[:300]}"
            )
        mem_section = "\n".join(mem_parts)
    else:
        mem_section = "## Relevant Past Iterations\n(none retrieved)"
    sections.append(_truncate(mem_section, memory_chars, "memory"))

    # --- Section 4: Rolling self-summary ---
    if rolling_summary:
        summary_section = f"## Progress Summary (this subtask so far)\n{rolling_summary}"
    else:
        summary_section = "## Progress Summary\n(starting fresh)"
    sections.append(_truncate(summary_section, summary_chars, "rolling_summary"))

    result = "\n\n".join(sections)
    total_chars = len(result)
    total_tokens_est = total_chars // CHARS_PER_TOKEN
    logger.debug(
        "Subagent context assembled: %d chars (~%d tokens), budget: %d tokens",
        total_chars,
        total_tokens_est,
        token_budget,
    )
    return result


def regenerate_rolling_summary_prompt(
    brief: SubtaskBrief,
    steps_history: list[dict],
) -> str:
    """Build the prompt asking the model to summarize progress so far."""
    steps_text = "\n".join(
        f"Step {i+1}: {s.get('action', '')} → {str(s.get('result', ''))[:200]}"
        for i, s in enumerate(steps_history[-20:])
    )
    return (
        f"You are working on subtask: {brief.goal}\n\n"
        f"Steps taken so far:\n{steps_text}\n\n"
        "Write a concise 3-5 sentence summary of what has been accomplished, "
        "what files were changed, and what still needs to be done. "
        "Be specific about file names and changes. Do not repeat the goal."
    )
