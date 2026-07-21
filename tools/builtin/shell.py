"""Built-in shell execution tool.

Danger classification is handled by ToolRuntime BEFORE this subprocess runs,
so this file only needs to execute the (pre-approved) command.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from tools.decorator import tool


@tool(name="run_shell", description="Run a shell command in the workspace directory", kind="action")
def run_shell(workspace: str, command: str, timeout: int = 60) -> dict:
    """Execute a shell command with CWD set to the workspace root.

    Args:
        workspace: auto-injected — do NOT pass.
        command: Shell command string.
        timeout: Max seconds to wait (default 60).

    Returns:
        {"stdout": str, "stderr": str, "returncode": int}

    Dangerous commands (rm, curl, git push, etc.) require user permission and are
    intercepted before reaching this function.
    """
    ws = Path(workspace)
    if not ws.exists():
        return {"stdout": "", "stderr": f"workspace not found: {workspace}", "returncode": 1}

    try:
        result = subprocess.run(
            command,
            shell=True,
            cwd=str(ws),
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return {
            "stdout": result.stdout[:16384],
            "stderr": result.stderr[:4096],
            "returncode": result.returncode,
        }
    except subprocess.TimeoutExpired:
        return {"stdout": "", "stderr": f"command timed out after {timeout}s", "returncode": -1}
    except Exception as exc:
        return {"stdout": "", "stderr": str(exc), "returncode": -1}
