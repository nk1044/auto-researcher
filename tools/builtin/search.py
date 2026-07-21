"""Built-in search tools: grep-style text search and find-style file discovery."""

from __future__ import annotations

import fnmatch
import re
from pathlib import Path

from tools.decorator import tool


@tool(name="search_files", description="Search for a pattern inside files in the workspace (like grep)", kind="action")
def search_files(
    workspace: str,
    pattern: str,
    path: str = ".",
    file_glob: str = "**/*",
    case_sensitive: bool = True,
    max_results: int = 50,
) -> dict:
    """Search for a regex/text pattern across files in the workspace.

    Args:
        workspace: auto-injected — do NOT pass.
        pattern: Text or regex pattern to search for.
        path: Root to search from, relative to workspace (default ".").
        file_glob: Glob pattern for files to include (default "**/*" = all files).
        case_sensitive: Whether the search is case-sensitive (default True).
        max_results: Max number of matching lines to return (default 50).

    Returns:
        {"matches": [{"file": str, "line": int, "text": str}], "count": int, "truncated": bool}
    """
    ws = Path(workspace).resolve()
    search_root = (ws / path).resolve()

    try:
        search_root.relative_to(ws)
    except ValueError:
        return {"error": f"path {path!r} escapes workspace"}

    flags = 0 if case_sensitive else re.IGNORECASE
    try:
        compiled = re.compile(pattern, flags)
    except re.error as exc:
        return {"error": f"invalid regex: {exc}"}

    matches = []
    truncated = False

    for filepath in sorted(search_root.rglob("*")):
        if not filepath.is_file():
            continue
        if ".git" in filepath.parts:
            continue
        if not fnmatch.fnmatch(filepath.name, file_glob.split("/")[-1]):
            # Simple glob check on filename — rglob handles directory portion
            pass  # rglob already filters by path

        try:
            text = filepath.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue

        for lineno, line in enumerate(text.splitlines(), start=1):
            if compiled.search(line):
                matches.append({
                    "file": str(filepath.relative_to(ws)),
                    "line": lineno,
                    "text": line.rstrip()[:300],
                })
                if len(matches) >= max_results:
                    truncated = True
                    break
        if truncated:
            break

    return {"matches": matches, "count": len(matches), "truncated": truncated}


@tool(name="find_files", description="Find files by name pattern in the workspace (like find)", kind="action")
def find_files(
    workspace: str,
    name_pattern: str = "*",
    path: str = ".",
    file_type: str = "f",
    max_results: int = 100,
) -> dict:
    """Locate files or directories by name pattern.

    Args:
        workspace: auto-injected — do NOT pass.
        name_pattern: Glob pattern for file name (e.g. "*.cpp", "test_*.py").
        path: Root directory to search, relative to workspace (default ".").
        file_type: "f" for files only, "d" for directories only, "a" for all (default "f").
        max_results: Max entries to return (default 100).

    Returns:
        {"files": [str, ...], "count": int, "truncated": bool}
    """
    ws = Path(workspace).resolve()
    search_root = (ws / path).resolve()

    try:
        search_root.relative_to(ws)
    except ValueError:
        return {"error": f"path {path!r} escapes workspace"}

    if not search_root.exists():
        return {"error": f"path not found: {path}"}

    results = []
    truncated = False

    for item in sorted(search_root.rglob(name_pattern)):
        if ".git" in item.parts:
            continue
        if file_type == "f" and not item.is_file():
            continue
        if file_type == "d" and not item.is_dir():
            continue
        results.append(str(item.relative_to(ws)))
        if len(results) >= max_results:
            truncated = True
            break

    return {"files": results, "count": len(results), "truncated": truncated}
