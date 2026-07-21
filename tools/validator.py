from __future__ import annotations

import fnmatch
import re


_DEFAULT_PROTECTED: list[str] = [
    "test/",
    "tests/",
    "*.test.*",
    "held_out/",
    "eval/",
    "benchmark/",
]

_DIFF_HEADER_RE = re.compile(r"^(?:\+\+\+|---)\s+(.+?)(?:\t.*)?$", re.MULTILINE)


def extract_touched_files(diff: str) -> list[str]:
    paths: list[str] = []
    for match in _DIFF_HEADER_RE.finditer(diff):
        raw = match.group(1).strip()
        if raw == "/dev/null":
            continue
        path = re.sub(r"^[ab]/", "", raw)
        if path and path not in paths:
            paths.append(path)
    return paths


def _matches_any(path: str, patterns: list[str]) -> bool:
    for pattern in patterns:
        if fnmatch.fnmatch(path, pattern):
            return True
        if pattern.endswith("/") and (path.startswith(pattern) or f"/{pattern}" in path):
            return True
        if not pattern.endswith("/") and not any(c in pattern for c in ("*", "?", "[")):
            if path.startswith(pattern) or f"/{pattern}" in f"/{path}":
                return True
    return False


def validate_diff(
    diff: str,
    repo_path: str,
    protected_patterns: list[str],
) -> tuple[bool, str]:
    effective_protected = protected_patterns if protected_patterns else _DEFAULT_PROTECTED

    if not diff or not diff.strip():
        return False, "diff is empty"

    touched = extract_touched_files(diff)
    if not touched:
        return False, "diff touches no recognisable files"

    for path in touched:
        if _matches_any(path, effective_protected):
            return False, f"diff modifies protected path: {path}"

    hunk_header = re.compile(r"^@@\s+-\d+(?:,\d+)?\s+\+\d+(?:,\d+)?\s+@@", re.MULTILINE)
    if not hunk_header.search(diff):
        return False, "diff contains no valid hunk headers — not a parseable unified diff"

    return True, ""
