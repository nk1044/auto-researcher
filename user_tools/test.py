"""Evaluation tool: runs the project's main.py via uv and returns a score based on final val loss.

This file is auto-discovered by ToolRuntime from the user_tools/ directory.
The only contract: return {"score": float, "remark": str | null}.

Score formula: 1.0 / (1.0 + final_val_loss)  →  lower loss = higher score, range (0, 1].
"""

from __future__ import annotations

import json
import math
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

from tools.decorator import tool

# Absolute path to the project directory (where main.py lives and the venv was created).
# The agent runs code in a git worktree, but we use THIS project's venv so that
# packages (torch, etc.) are available without recreating the environment in every worktree.
DEFAULT_PROJECT_DIR = "/home/arc2/vscode/moe-implementation"

TIMEOUT_SECONDS = 600


def _run_and_get_final_loss(project_dir: str, timeout: int = TIMEOUT_SECONDS) -> dict:
    """Run main.py inside project_dir and return the final train/val losses.

    Uses the project's own venv python (DEFAULT_PROJECT_DIR/.venv/bin/python) so
    that git worktrees — which share the repo files but not the venv — still have
    access to all installed packages without triggering a uv venv recreation.

    Returns:
        {"final_train_loss": float, "final_val_loss": float, "stdout": str}
    Raises:
        RuntimeError if the process fails or no [eval] lines are found.
    """
    # Copy gitignored data files that the project needs but aren't in the worktree.
    # input.txt is the training corpus — it's in .gitignore so git worktree add skips it.
    src_root = Path(DEFAULT_PROJECT_DIR)
    dst_root = Path(project_dir)
    for data_file in ["input.txt"]:
        src = src_root / data_file
        dst = dst_root / data_file
        if src.exists() and not dst.exists():
            shutil.copy2(str(src), str(dst))

    python_exe = src_root / ".venv" / "bin" / "python"
    if python_exe.exists():
        cmd = [str(python_exe), "main.py"]
    else:
        # Fallback: let uv figure it out (only works when running directly in the project)
        cmd = ["uv", "run", "main.py"]

    # Strip VIRTUAL_ENV so the subprocess isn't confused by the auto-researcher's venv
    env = {k: v for k, v in os.environ.items() if k != "VIRTUAL_ENV"}

    result = subprocess.run(
        cmd,
        cwd=project_dir,
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
    )

    stdout = result.stdout
    stderr = result.stderr

    if result.returncode != 0:
        raise RuntimeError(
            f"main.py exited with code {result.returncode}.\n"
            f"--- stderr ---\n{stderr}\n--- stdout ---\n{stdout}"
        )

    # Parse all [eval] lines: "[eval]  step  NNN: train X.XXXX  val Y.YYYY"
    pattern = re.compile(
        r"\[eval\]\s+step\s+\d+:\s+train\s+([0-9.]+)\s+val\s+([0-9.]+)"
    )
    matches = pattern.findall(stdout)

    if not matches:
        raise RuntimeError(
            f"No [eval] lines found in output.\n--- stdout ---\n{stdout}"
        )

    final_train, final_val = matches[-1]
    return {
        "final_train_loss": float(final_train),
        "final_val_loss": float(final_val),
        "stdout": stdout,
    }


@tool(name="run_tests", description="Run the project training via uv and return a score based on final val loss", kind="test")
def run_tests(workspace: str) -> dict:
    """Run `uv run main.py` in the project workspace and return a loss-based score.

    Args:
        workspace: Absolute path to the project directory to evaluate.
                   Falls back to DEFAULT_PROJECT_DIR if not provided.

    Returns:
        {"score": float in (0, 1], "remark": str}
        Score = 1.0 / (1.0 + final_val_loss) — lower loss gives higher score.
    """
    project_dir = workspace if workspace else DEFAULT_PROJECT_DIR

    if not Path(project_dir).exists():
        return {"score": 0.0, "remark": f"project directory not found: {project_dir}"}

    try:
        losses = _run_and_get_final_loss(project_dir)
        train_loss = losses["final_train_loss"]
        val_loss = losses["final_val_loss"]

        if not math.isfinite(val_loss) or val_loss < 0:
            return {"score": 0.0, "remark": f"invalid val_loss={val_loss} (nan/inf/negative) — treating as failure"}

        # Map val loss to (0, 1]: lower loss → score closer to 1
        score = 1.0 / (1.0 + val_loss)

        remark = (
            f"final_train_loss={train_loss:.4f}  final_val_loss={val_loss:.4f}  "
            f"score={score:.4f}"
        )
        return {"score": score, "remark": remark}

    except subprocess.TimeoutExpired:
        return {"score": 0.0, "remark": f"run timed out after {TIMEOUT_SECONDS}s"}
    except RuntimeError as exc:
        return {"score": 0.0, "remark": str(exc)}
    except Exception as exc:
        return {"score": 0.0, "remark": f"unexpected error: {exc}"}


# When run as the sandbox subprocess entry point
if __name__ == "__main__":
    kwargs = json.loads(sys.stdin.read())
    try:
        value = run_tests(**kwargs)
        print(json.dumps({"success": True, "value": value}))
    except Exception as e:
        print(json.dumps({"success": False, "error": str(e)}))
