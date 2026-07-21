from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from coordinator.integrator import apply_diff_to_worktree, create_integration_worktree, naive_merge
from shared.types import SubtaskResult, TaskStatus


def _init_repo(path: Path) -> None:
    subprocess.run(["git", "init", str(path)], check=True, capture_output=True)
    subprocess.run(
        ["git", "-C", str(path), "config", "user.email", "test@test.com"],
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "-C", str(path), "config", "user.name", "Test"],
        check=True,
        capture_output=True,
    )


def _commit(repo: Path) -> str:
    subprocess.run(
        ["git", "-C", str(repo), "add", "-A"], check=True, capture_output=True
    )
    subprocess.run(
        ["git", "-C", str(repo), "commit", "--allow-empty", "-m", "setup"],
        check=True,
        capture_output=True,
    )
    return subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


async def test_naive_merge_non_overlapping_both_accepted(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _init_repo(repo)

    (repo / "a.py").write_text("x = 1\nextra = 0\n")
    (repo / "b.py").write_text("p = 3\nextra = 0\n")
    commit = _commit(repo)

    worktree_root = str(tmp_path / "wts")
    integration_wt = await create_integration_worktree(commit, str(repo), worktree_root, "merge-1")

    diff_a = (
        "--- a/a.py\n"
        "+++ b/a.py\n"
        "@@ -1,2 +1,2 @@\n"
        "-x = 1\n"
        "+x = 10\n"
        " extra = 0\n"
    )
    diff_b = (
        "--- a/b.py\n"
        "+++ b/b.py\n"
        "@@ -1,2 +1,2 @@\n"
        "-p = 3\n"
        "+p = 30\n"
        " extra = 0\n"
    )

    results = [
        SubtaskResult(subtask_id="s1", diff=diff_a, status=TaskStatus.SUCCESS),
        SubtaskResult(subtask_id="s2", diff=diff_b, status=TaskStatus.SUCCESS),
    ]

    final_diff, accepted, rejected = await naive_merge(results, integration_wt)

    assert "s1" in accepted
    assert "s2" in accepted
    assert rejected == []

    merged_a = (Path(integration_wt) / "a.py").read_text()
    merged_b = (Path(integration_wt) / "b.py").read_text()
    assert "x = 10" in merged_a
    assert "p = 30" in merged_b

    await _remove_wt(integration_wt, str(repo))


async def test_naive_merge_overlapping_first_accepted_second_rejected(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _init_repo(repo)

    (repo / "c.py").write_text("val = 42\nextra = 0\n")
    commit = _commit(repo)

    worktree_root = str(tmp_path / "wts")
    integration_wt = await create_integration_worktree(commit, str(repo), worktree_root, "merge-2")

    diff_1 = (
        "--- a/c.py\n"
        "+++ b/c.py\n"
        "@@ -1,2 +1,2 @@\n"
        "-val = 42\n"
        "+val = 100\n"
        " extra = 0\n"
    )
    # Diff 2 also rewrites the original line — conflicts after diff 1 is applied.
    diff_2 = (
        "--- a/c.py\n"
        "+++ b/c.py\n"
        "@@ -1,2 +1,2 @@\n"
        "-val = 42\n"
        "+val = 200\n"
        " extra = 0\n"
    )

    results = [
        SubtaskResult(subtask_id="s1", diff=diff_1, status=TaskStatus.SUCCESS),
        SubtaskResult(subtask_id="s2", diff=diff_2, status=TaskStatus.SUCCESS),
    ]

    final_diff, accepted, rejected = await naive_merge(results, integration_wt)

    assert "s1" in accepted
    assert "s2" in rejected

    content = (Path(integration_wt) / "c.py").read_text()
    assert "val = 100" in content
    assert "val = 200" not in content

    await _remove_wt(integration_wt, str(repo))


async def test_naive_merge_failed_subtask_rejected(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _init_repo(repo)
    commit = _commit(repo)

    worktree_root = str(tmp_path / "wts")
    integration_wt = await create_integration_worktree(commit, str(repo), worktree_root, "merge-3")

    results = [
        SubtaskResult(subtask_id="fail-1", diff="", status=TaskStatus.FAILED),
        SubtaskResult(subtask_id="fail-2", diff=None, status=TaskStatus.FAILED),
    ]

    _, accepted, rejected = await naive_merge(results, integration_wt)

    assert accepted == []
    assert "fail-1" in rejected
    assert "fail-2" in rejected

    await _remove_wt(integration_wt, str(repo))


async def test_apply_diff_valid_diff_returns_true(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _init_repo(repo)

    (repo / "target.py").write_text("answer = 41\nextra = 0\n")
    commit = _commit(repo)

    worktree_root = str(tmp_path / "wts")
    wt = await create_integration_worktree(commit, str(repo), worktree_root, "apply-1")

    valid_diff = (
        "--- a/target.py\n"
        "+++ b/target.py\n"
        "@@ -1,2 +1,2 @@\n"
        "-answer = 41\n"
        "+answer = 42\n"
        " extra = 0\n"
    )

    result = await apply_diff_to_worktree(valid_diff, wt)

    assert result is True
    assert "answer = 42" in (Path(wt) / "target.py").read_text()

    await _remove_wt(wt, str(repo))


async def test_apply_diff_invalid_diff_returns_false_no_exception(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _init_repo(repo)
    commit = _commit(repo)

    worktree_root = str(tmp_path / "wts")
    wt = await create_integration_worktree(commit, str(repo), worktree_root, "apply-2")

    malformed = "this is definitely not a valid unified diff format\nrandom text here"

    result = await apply_diff_to_worktree(malformed, wt)

    assert result is False

    await _remove_wt(wt, str(repo))


async def test_apply_diff_empty_diff_returns_true(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _init_repo(repo)
    commit = _commit(repo)

    worktree_root = str(tmp_path / "wts")
    wt = await create_integration_worktree(commit, str(repo), worktree_root, "apply-3")

    result = await apply_diff_to_worktree("", wt)
    assert result is True

    result_ws = await apply_diff_to_worktree("   \n  ", wt)
    assert result_ws is True

    await _remove_wt(wt, str(repo))


async def _remove_wt(path: str, repo: str) -> None:
    from coordinator.integrator import remove_worktree
    await remove_worktree(path, repo)
