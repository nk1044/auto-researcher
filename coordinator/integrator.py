"""Integration step: coordinator reviews and merges subagent results."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import subprocess
import tempfile
import uuid
from pathlib import Path
from typing import Any

from shared.types import IntegrationResult, SubtaskResult, TaskStatus

logger = logging.getLogger(__name__)

REVIEW_SYSTEM = """You are a senior software engineer reviewing parallel subagent contributions to a single hypothesis.

Your job:
1. Review each subagent's diff.
2. Decide: ACCEPT, PATCH (describe fix), or REJECT (with reason) for each subtask.
3. Produce a SINGLE unified diff that merges all accepted/patched changes.

The merged diff must:
- Be a valid unified diff applicable with `git apply`.
- Not break any imports or cross-references.
- Not repeat changes (if two subtasks touched the same lines, resolve conflicts).

Output JSON:
{
  "decisions": [
    {"subtask_id": "...", "decision": "ACCEPT|PATCH|REJECT", "reason": "..."}
  ],
  "merged_diff": "...unified diff string or empty string if nothing to merge...",
  "summary": "...what the merged change accomplishes..."
}
"""

REVIEW_USER = """Hypothesis: {hypothesis}

Subagent Results:
{results_text}

Review the diffs above and produce a merged integration.
If all diffs are empty or all subtasks failed, output an empty merged_diff."""


def _extract_json(text: str) -> dict:
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
    raise ValueError(f"Cannot extract JSON from integration response: {text[:300]}")


def _format_results_for_review(results: list[SubtaskResult]) -> str:
    parts: list[str] = []
    for r in results:
        diff_preview = r.diff[:1000] if r.diff else "(no changes)"
        parts.append(
            f"--- Subtask {r.subtask_id} (status={r.status.value}) ---\n"
            f"Summary: {r.summary}\n"
            f"Files: {', '.join(r.files_touched) if r.files_touched else 'none'}\n"
            f"Diff:\n```\n{diff_preview}\n```"
        )
    return "\n\n".join(parts)


async def apply_diff_to_worktree(diff: str, worktree_path: str) -> bool:
    """Apply a unified diff to a git worktree. Returns True on success."""
    if not diff.strip():
        return True

    proc = await asyncio.create_subprocess_exec(
        "git", "apply", "--index", "-",
        cwd=worktree_path,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await proc.communicate(input=diff.encode())
    if proc.returncode != 0:
        logger.warning("git apply failed in %s: %s", worktree_path, stderr.decode()[:500])
        return False
    return True


async def create_integration_worktree(
    baseline_commit: str,
    repo_path: str,
    worktree_root: str,
    integration_id: str,
) -> str:
    """Create a fresh worktree for the integration result."""
    os.makedirs(worktree_root, exist_ok=True)
    path = os.path.join(worktree_root, f"integration-{integration_id}")

    proc = await asyncio.create_subprocess_exec(
        "git", "worktree", "add", "--detach", path, baseline_commit,
        cwd=repo_path,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await proc.communicate()
    if proc.returncode != 0:
        raise RuntimeError(f"Failed to create integration worktree: {stderr.decode()}")
    return path


async def get_worktree_diff(worktree_path: str) -> str:
    """Get the full diff of changes in a worktree vs HEAD."""
    proc = await asyncio.create_subprocess_exec(
        "git", "diff", "HEAD",
        cwd=worktree_path,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, _ = await proc.communicate()
    return stdout.decode(errors="replace")


async def remove_worktree(worktree_path: str, repo_path: str) -> None:
    """Remove a git worktree."""
    proc = await asyncio.create_subprocess_exec(
        "git", "worktree", "remove", "--force", worktree_path,
        cwd=repo_path,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    await proc.communicate()


def build_integration_messages(
    hypothesis_text: str,
    results: list[SubtaskResult],
) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": REVIEW_SYSTEM},
        {
            "role": "user",
            "content": REVIEW_USER.format(
                hypothesis=hypothesis_text,
                results_text=_format_results_for_review(results),
            ),
        },
    ]


async def naive_merge(
    results: list[SubtaskResult],
    integration_worktree: str,
) -> tuple[str, list[str], list[str]]:
    """Fallback: apply each non-empty diff sequentially, skip on conflict."""
    accepted: list[str] = []
    rejected: list[str] = []

    for r in results:
        if not r.diff or r.status == TaskStatus.FAILED:
            rejected.append(r.subtask_id)
            continue
        ok = await apply_diff_to_worktree(r.diff, integration_worktree)
        if ok:
            accepted.append(r.subtask_id)
        else:
            rejected.append(r.subtask_id)
            logger.warning("Naive merge: subtask %s diff rejected (conflict)", r.subtask_id)

    final_diff = await get_worktree_diff(integration_worktree)
    return final_diff, accepted, rejected
