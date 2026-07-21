from __future__ import annotations

import asyncio
import logging
from typing import Any

from tools.decorator import tool

logger = logging.getLogger(__name__)


async def _git(
    *args: str,
    cwd: str,
    input: bytes | None = None,
) -> tuple[int, str, str]:
    proc = await asyncio.create_subprocess_exec(
        "git",
        *args,
        cwd=cwd,
        stdin=asyncio.subprocess.PIPE if input is not None else asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await proc.communicate(input=input)
    return proc.returncode, stdout.decode(), stderr.decode()


@tool(name="save_to_github", description="Commit integrated diff and push to GitHub", kind="save")
async def save_to_github(
    diff: str,
    repo_path: str,
    remote: str,
    branch_prefix: str,
    iteration: int,
    meta: dict[str, Any],
) -> dict[str, str]:
    branch_name = f"{branch_prefix}/{iteration:04d}"

    rc, _, err = await _git("checkout", "-B", branch_name, cwd=repo_path)
    if rc != 0:
        raise RuntimeError(f"git checkout failed: {err.strip()}")

    rc, _, err = await _git(
        "apply", "--index",
        cwd=repo_path,
        input=diff.encode(),
    )
    if rc != 0:
        raise RuntimeError(f"git apply failed: {err.strip()}")

    hypothesis = meta.get("hypothesis", "")
    score = meta.get("score", "")
    commit_msg_parts = ["auto-researcher: apply diff"]
    if hypothesis:
        short_hyp = hypothesis[:120].replace("\n", " ")
        commit_msg_parts.append(f"hypothesis: {short_hyp}")
    if score != "":
        commit_msg_parts.append(f"score: {score}")
    commit_msg = "\n\n".join(commit_msg_parts)

    rc, _, err = await _git(
        "commit", "--no-gpg-sign", "-m", commit_msg,
        cwd=repo_path,
    )
    if rc != 0:
        raise RuntimeError(f"git commit failed: {err.strip()}")

    rc, sha_out, err = await _git("rev-parse", "HEAD", cwd=repo_path)
    if rc != 0:
        raise RuntimeError(f"git rev-parse failed: {err.strip()}")
    sha = sha_out.strip()

    rc, _, err = await _git("push", remote, branch_name, cwd=repo_path)
    if rc != 0:
        raise RuntimeError(f"git push failed: {err.strip()}")

    remote_url = remote
    rc, url_out, _ = await _git("remote", "get-url", remote, cwd=repo_path)
    if rc == 0:
        remote_url = url_out.strip()

    logger.info("pushed branch %s (%s) to %s", branch_name, sha[:12], remote_url)
    return {"branch": branch_name, "commit": sha, "url": remote_url}
