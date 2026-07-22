"""Built-in file CRUD tools — workspace-scoped, path-safe."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from tools.decorator import tool


# ── Path safety ────────────────────────────────────────────────────────────────

def _safe_resolve(workspace: str, rel_path: str) -> tuple[Path | None, str]:
    """Resolve rel_path inside workspace; return (path, error_msg)."""
    ws = Path(workspace).resolve()
    try:
        target = (ws / rel_path).resolve()
        target.relative_to(ws)   # raises ValueError if outside
        return target, ""
    except ValueError:
        return None, f"path {rel_path!r} escapes workspace"


# ── Tools ───────────────────────────────────────────────────────────────────────

@tool(name="read_file", description="Read a file from the workspace with optional line range", kind="action")
def read_file(workspace: str, path: str, offset: int = 0, limit: int = 0) -> dict:
    """Read a file from the workspace. Lines are shown with line numbers.

    Args:
        workspace: auto-injected — do NOT pass.
        path: File path relative to workspace root (e.g. "src/main.cpp").
        offset: First line to read (0-based, default 0).
        limit: Max lines to return (0 = all).

    Returns:
        {"content": str, "total_lines": int, "shown": str, "path": str}
    """
    target, err = _safe_resolve(workspace, path)
    if err:
        return {"error": err}
    if not target.exists():
        return {"error": f"file not found: {path}"}
    if not target.is_file():
        return {"error": f"not a file: {path}"}

    raw = target.read_text(encoding="utf-8", errors="replace")
    lines = raw.splitlines(keepends=True)
    total = len(lines)

    start = max(0, offset)
    end = (start + limit) if limit > 0 else total
    end = min(end, total)

    chunk = lines[start:end]
    numbered = "".join(f"{start + i + 1:>5}\t{ln}" for i, ln in enumerate(chunk))

    return {
        "content": numbered,
        "total_lines": total,
        "shown": f"{start + 1}-{end}",
        "path": str(target.relative_to(Path(workspace).resolve())),
    }


@tool(name="write_file", description="Create or overwrite a file in the workspace", kind="action")
def write_file(workspace: str, path: str, content: str) -> dict:
    """Write the complete content to a file. Creates parent directories as needed.

    Args:
        workspace: auto-injected — do NOT pass.
        path: File path relative to workspace root.
        content: Complete file content (do NOT truncate).

    Returns:
        {"written": str, "bytes": int, "lines": int}
    """
    target, err = _safe_resolve(workspace, path)
    if err:
        return {"error": err}

    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    ws = Path(workspace).resolve()
    return {
        "written": str(target.relative_to(ws)),
        "bytes": len(content.encode("utf-8")),
        "lines": content.count("\n") + 1,
    }


@tool(name="edit_file", description="Replace an exact string in a file — surgical edit without rewriting the whole file", kind="action")
def edit_file(workspace: str, path: str, old_string: str, new_string: str) -> dict:
    """Find old_string in file and replace it with new_string.

    Fails clearly if old_string appears 0 or more than 1 time.
    Prefer this over write_file for targeted changes — far less risk of
    accidentally dropping unrelated code.

    Args:
        workspace: auto-injected — do NOT pass.
        path: File path relative to workspace root.
        old_string: Exact string to find (must be unique in the file).
        new_string: Replacement string.

    Returns:
        {"patched": str, "old_lines": int, "new_lines": int}
    """
    target, err = _safe_resolve(workspace, path)
    if err:
        return {"error": err}
    if not target.exists():
        return {"error": f"file not found: {path}"}

    text = target.read_text(encoding="utf-8", errors="replace")
    count = text.count(old_string)

    if count == 0:
        # Show first 800 chars of file to help model understand what's actually there
        preview = text[:800] + ("…" if len(text) > 800 else "")
        return {
            "error": f"old_string not found in {path}.",
            "hint": "The string you specified does not exist verbatim. Read the file first and copy the exact characters.",
            "file_preview": preview,
        }
    if count > 1:
        return {
            "error": f"old_string appears {count} times — make it more specific by including more surrounding context.",
        }

    new_text = text.replace(old_string, new_string, 1)
    target.write_text(new_text, encoding="utf-8")
    ws = Path(workspace).resolve()
    patched = str(target.relative_to(ws))
    return {
        "written": patched,              # matches files_touched tracker in subagent.py
        "patched": patched,
        "old_lines": old_string.count("\n") + 1,
        "new_lines": new_string.count("\n") + 1,
    }


@tool(name="list_files", description="List files and directories in the workspace", kind="action")
def list_files(workspace: str, path: str = ".", pattern: str = "") -> dict:
    """List contents of a directory in the workspace.

    Args:
        workspace: auto-injected — do NOT pass.
        path: Directory to list, relative to workspace root (default: ".").
        pattern: Optional glob pattern (e.g. "*.cpp", "**/*.py", "src/**").

    Returns:
        {"entries": [{"path": str, "type": "file"|"dir", "size": int}], "count": int}
    """
    target, err = _safe_resolve(workspace, path)
    if err:
        return {"error": err}
    if not target.exists():
        return {"error": f"path not found: {path}"}

    ws = Path(workspace).resolve()

    if pattern:
        items = sorted(target.glob(pattern))
    elif target.is_dir():
        items = sorted(target.iterdir())
    else:
        items = [target]

    entries = []
    for item in items[:300]:
        if ".git" in item.parts:
            continue
        try:
            rel = str(item.relative_to(ws))
            kind = "dir" if item.is_dir() else "file"
            size = item.stat().st_size if item.is_file() else 0
            entries.append({"path": rel, "type": kind, "size": size})
        except (ValueError, OSError):
            pass

    return {"entries": entries, "count": len(entries)}


@tool(name="delete_file", description="Delete a file from the workspace", kind="action")
def delete_file(workspace: str, path: str) -> dict:
    """Delete a single file from the workspace.

    Args:
        workspace: auto-injected — do NOT pass.
        path: File path relative to workspace root.

    Returns:
        {"deleted": str}
    """
    target, err = _safe_resolve(workspace, path)
    if err:
        return {"error": err}
    if not target.exists():
        return {"error": f"not found: {path}"}
    if target.is_dir():
        return {"error": f"{path!r} is a directory. Use run_shell with 'rm -rf' (requires permission)."}

    target.unlink()
    ws = Path(workspace).resolve()
    return {"deleted": str(target.relative_to(ws))}


@tool(name="create_dir", description="Create a directory in the workspace", kind="action")
def create_dir(workspace: str, path: str) -> dict:
    """Create a directory (and any missing parents) in the workspace.

    Args:
        workspace: auto-injected — do NOT pass.
        path: Directory path relative to workspace root.

    Returns:
        {"created": str}
    """
    target, err = _safe_resolve(workspace, path)
    if err:
        return {"error": err}

    target.mkdir(parents=True, exist_ok=True)
    ws = Path(workspace).resolve()
    return {"created": str(target.relative_to(ws))}
