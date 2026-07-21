"""Sample action tool: runs a shell command in the workspace.

This file is auto-discovered by ToolRuntime. It demonstrates how to register
a custom action tool that subagents and the coordinator can call.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from tools.decorator import tool


@tool(name="run_shell", description="Run a shell command in workspace", kind="action")
def run_shell(workspace: str, command: str) -> dict:
    """Run a shell command inside the workspace directory.

    Args:
        workspace: Absolute path to the git worktree.
        command: Shell command to execute.

    Returns:
        {"stdout": str, "stderr": str, "returncode": int}
    """
    repo = Path(workspace)
    if not repo.exists():
        return {"stdout": "", "stderr": f"workspace not found: {workspace}", "returncode": 1}

    result = subprocess.run(
        command,
        shell=True,
        cwd=str(repo),
        capture_output=True,
        text=True,
        timeout=60,
    )
    return {
        "stdout": result.stdout[:4096],
        "stderr": result.stderr[:2048],
        "returncode": result.returncode,
    }


if __name__ == "__main__":
    kwargs = json.loads(sys.stdin.read())
    try:
        value = run_shell(**kwargs)
        print(json.dumps({"success": True, "value": value}))
    except Exception as e:
        print(json.dumps({"success": False, "error": str(e)}))
