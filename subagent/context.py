"""Context assembly for subagents: bounded, deterministic, never accumulating."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

from shared.types import MemoryEntry, SubtaskBrief

logger = logging.getLogger(__name__)

CHARS_PER_TOKEN = 3


def _budget_chars(token_budget: int, ratio: float) -> int:
    return int(token_budget * ratio * CHARS_PER_TOKEN)


def _truncate(text: str, max_chars: int, label: str) -> str:
    if len(text) <= max_chars:
        return text
    keep = max_chars - 40
    logger.debug("Section %r truncated from %d to %d chars", label, len(text), max_chars)
    return text[:keep] + f"\n... [truncated — {len(text) - keep} chars omitted]"


def _list_workspace_files(workspace: str) -> list[str]:
    """Return all non-.git files in the workspace, relative paths."""
    ws = Path(workspace)
    files = []
    for p in sorted(ws.rglob("*")):
        if p.is_file() and ".git" not in p.parts:
            files.append(str(p.relative_to(ws)))
    return files


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
    workspace: Optional[str] = None,
) -> str:
    task_brief_chars = _budget_chars(token_budget, ratios.get("task_brief", 0.20))
    files_chars      = _budget_chars(token_budget, ratios.get("files", 0.50))
    memory_chars     = _budget_chars(token_budget, ratios.get("memory", 0.15))
    summary_chars    = _budget_chars(token_budget, ratios.get("rolling_summary", 0.15))

    sections: list[str] = []

    # --- Section 1: Task brief + workspace listing + tool schemas ---
    brief_text = (
        f"## Subtask Brief\n"
        f"Goal: {brief.goal}\n"
        f"Suggested scope: {', '.join(brief.scope) if brief.scope else 'none — explore workspace'}\n"
        f"Constraints: {brief.constraints}\n"
        f"Expected output: {brief.expected_output}\n"
    )

    if workspace:
        ws_files = _list_workspace_files(workspace)
        listing = "\n".join(ws_files[:100])
        brief_text += (
            f"\n## Actual Workspace Path\n"
            f"{workspace}\n"
            f"(This is an isolated git worktree — a full copy of the target repo at the baseline commit. "
            f"Edit files here freely; your changes will be tested and applied if they improve the score.)\n"
            f"\n## Actual Workspace Files (USE THESE EXACT RELATIVE PATHS — do NOT use absolute paths)\n"
            f"{listing}\n"
            f"\nNOTE: The suggested scope above may be inaccurate. "
            f"Always use file paths from the list above."
        )

    if tool_schemas:
        brief_text += "\n\n## Available Tools\n" + json.dumps(tool_schemas, indent=2)

    sections.append(_truncate(brief_text, task_brief_chars, "task_brief"))

    # --- Section 2: In-scope file slices (resolved relative to workspace) ---
    scope_paths = brief.scope if brief.scope else []
    if workspace and scope_paths:
        # Resolve each scope path relative to the workspace
        resolved: list[tuple[str, str]] = []
        ws_files_set = set(ws_files) if workspace else set()
        for rel_path in scope_paths:
            full = str(Path(workspace) / rel_path)
            if rel_path in ws_files_set:
                resolved.append((rel_path, full))
            else:
                # Try to find the closest match by filename
                name = Path(rel_path).name
                matches = [f for f in ws_files_set if Path(f).name == name]
                if matches:
                    match = matches[0]
                    resolved.append((match, str(Path(workspace) / match)))
                else:
                    resolved.append((rel_path, full))  # will show NOT FOUND

        per_file_chars = max(200, files_chars // max(len(resolved), 1))
        file_parts: list[str] = ["## In-Scope File Contents"]
        for rel_path, full_path in resolved:
            content = read_file_slice(full_path, per_file_chars)
            if content.startswith("[could not read"):
                file_parts.append(
                    f"\n### {rel_path}\n"
                    f"⚠️  NOT FOUND — pick a file from the workspace listing above instead."
                )
            else:
                file_parts.append(f"\n### {rel_path}\n```\n{content}\n```")
        file_section = "\n".join(file_parts)
    elif workspace:
        file_section = "## In-Scope Files\n(no scope specified — pick relevant files from the workspace listing above)"
    else:
        file_section = "## In-Scope Files\n(workspace path not available)"

    sections.append(_truncate(file_section, files_chars, "files"))

    # --- Section 3: Retrieved memory ---
    if memory_entries:
        mem_parts = ["## Relevant Past Iterations"]
        for entry in memory_entries:
            label = f"[{entry.outcome.value if hasattr(entry.outcome, 'value') else entry.outcome}]"
            remark = f" — {entry.remark}" if entry.remark else ""
            mem_parts.append(f"- {label} score={entry.score:.3f}{remark}: {entry.text[:300]}")
        mem_section = "\n".join(mem_parts)
    else:
        mem_section = "## Relevant Past Iterations\n(none yet)"
    sections.append(_truncate(mem_section, memory_chars, "memory"))

    # --- Section 4: Rolling summary ---
    summary_section = (
        f"## Progress So Far\n{rolling_summary}"
        if rolling_summary
        else "## Progress So Far\n(first step)"
    )
    sections.append(_truncate(summary_section, summary_chars, "rolling_summary"))

    result = "\n\n".join(sections)
    logger.debug(
        "Subagent context: %d chars (~%d tokens), budget %d tokens",
        len(result), len(result) // CHARS_PER_TOKEN, token_budget,
    )
    return result


def regenerate_rolling_summary_prompt(brief: SubtaskBrief, steps_history: list[dict]) -> str:
    steps_text = "\n".join(
        f"Step {i+1}: {s.get('action', '')} → {str(s.get('result', ''))[:200]}"
        for i, s in enumerate(steps_history[-20:])
    )
    return (
        f"Subtask: {brief.goal}\n\n"
        f"Steps so far:\n{steps_text}\n\n"
        "Write a 3-5 sentence summary: what was accomplished, which files were changed, "
        "and what still needs to be done. Be specific about file names."
    )
