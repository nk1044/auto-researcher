"""Sample opaque test tool: runs pytest on the target repo and returns a pass-rate score.

This file is auto-discovered by ToolRuntime from the user_tools/ directory.
Replace the body of `run_tests` with your own evaluation logic.
The only contract: return {"score": float, "remark": str | null}.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from tools.decorator import tool


@tool(name="run_tests", description="Run the test suite on workspace and return a score", kind="test")
def run_tests(workspace: str) -> dict:
    """Run pytest in `workspace` and return a score based on pass rate.

    Args:
        workspace: Absolute path to the git worktree to evaluate.

    Returns:
        {"score": float in [0, 1], "remark": str | null}
    """
    repo = Path(workspace)
    if not repo.exists():
        return {"score": 0.0, "remark": f"workspace not found: {workspace}"}

    try:
        result = subprocess.run(
            [sys.executable, "-m", "pytest", "--tb=no", "-q", "--no-header"],
            cwd=str(repo),
            capture_output=True,
            text=True,
            timeout=120,
        )
        output = result.stdout + result.stderr
        passed, failed, errors = _parse_pytest_output(output)
        total = passed + failed + errors
        if total == 0:
            return {"score": 0.0, "remark": "no tests found"}
        score = passed / total
        remark = f"{passed}/{total} tests passed"
        if failed > 0:
            remark += f", {failed} failed"
        if errors > 0:
            remark += f", {errors} errors"
        return {"score": score, "remark": remark}
    except subprocess.TimeoutExpired:
        return {"score": 0.0, "remark": "test run timed out"}
    except Exception as exc:
        return {"score": 0.0, "remark": str(exc)}


def _parse_pytest_output(output: str) -> tuple[int, int, int]:
    """Parse pytest summary line: '3 passed, 1 failed in 0.5s'."""
    import re

    passed = failed = errors = 0
    summary = re.search(r"(\d+) passed", output)
    if summary:
        passed = int(summary.group(1))
    summary = re.search(r"(\d+) failed", output)
    if summary:
        failed = int(summary.group(1))
    summary = re.search(r"(\d+) error", output)
    if summary:
        errors = int(summary.group(1))
    return passed, failed, errors


# When run as the sandbox subprocess entry point
if __name__ == "__main__":
    kwargs = json.loads(sys.stdin.read())
    try:
        value = run_tests(**kwargs)
        print(json.dumps({"success": True, "value": value}))
    except Exception as e:
        print(json.dumps({"success": False, "error": str(e)}))
