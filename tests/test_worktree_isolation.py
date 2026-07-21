from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from coordinator.integrator import create_integration_worktree, remove_worktree


def _init_repo(path: Path) -> str:
    """Create a minimal git repo and return the initial commit hash."""
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
    subprocess.run(
        ["git", "-C", str(path), "commit", "--allow-empty", "-m", "init"],
        check=True,
        capture_output=True,
    )
    result = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


async def test_two_worktrees_get_separate_paths(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    commit = _init_repo(repo)
    worktree_root = str(tmp_path / "worktrees")

    path1 = await create_integration_worktree(commit, str(repo), worktree_root, "id-alpha")
    path2 = await create_integration_worktree(commit, str(repo), worktree_root, "id-beta")

    assert path1 != path2
    assert Path(path1).exists()
    assert Path(path2).exists()

    await remove_worktree(path1, str(repo))
    await remove_worktree(path2, str(repo))


async def test_modifying_one_worktree_does_not_affect_other(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    commit = _init_repo(repo)
    worktree_root = str(tmp_path / "worktrees")

    path1 = await create_integration_worktree(commit, str(repo), worktree_root, "wt-1")
    path2 = await create_integration_worktree(commit, str(repo), worktree_root, "wt-2")

    sentinel = "isolation_check.py"
    (Path(path1) / sentinel).write_text("x = 1\n")

    assert (Path(path1) / sentinel).exists()
    assert not (Path(path2) / sentinel).exists()

    await remove_worktree(path1, str(repo))
    await remove_worktree(path2, str(repo))


async def test_worktrees_share_same_baseline_commit(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "placeholder").mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init", str(repo)], check=True, capture_output=True)
    subprocess.run(
        ["git", "-C", str(repo), "config", "user.email", "t@t.com"],
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "-C", str(repo), "config", "user.name", "T"],
        check=True,
        capture_output=True,
    )
    (repo / "src.py").write_text("val = 42\n")
    subprocess.run(["git", "-C", str(repo), "add", "src.py"], check=True, capture_output=True)
    subprocess.run(
        ["git", "-C", str(repo), "commit", "-m", "add src"],
        check=True,
        capture_output=True,
    )
    commit = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()

    worktree_root = str(tmp_path / "wts")
    path1 = await create_integration_worktree(commit, str(repo), worktree_root, "c1")
    path2 = await create_integration_worktree(commit, str(repo), worktree_root, "c2")

    assert (Path(path1) / "src.py").read_text() == "val = 42\n"
    assert (Path(path2) / "src.py").read_text() == "val = 42\n"

    await remove_worktree(path1, str(repo))
    await remove_worktree(path2, str(repo))


async def test_remove_worktree_cleans_up_directory(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    commit = _init_repo(repo)
    worktree_root = str(tmp_path / "worktrees")

    wt_path = await create_integration_worktree(commit, str(repo), worktree_root, "cleanup-test")
    assert Path(wt_path).exists()

    await remove_worktree(wt_path, str(repo))

    assert not Path(wt_path).exists()
