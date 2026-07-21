"""Classify shell commands by danger level."""

from __future__ import annotations
import re

# Patterns that are ALWAYS blocked — could damage the host system regardless of context.
_HARD_BLOCKED: list[tuple[str, str]] = [
    (r':\(\)\s*\{.*\|.*&.*\}', "fork bomb"),
    (r'\bdd\b.*\bof=/dev/', "raw disk write"),
    (r'\bmkfs\.', "disk format"),
    (r'\bshred\b.*\s/dev/', "disk wipe"),
    (r'(curl|wget)[^\n;|]{0,200}[|>]\s*(bash|sh|zsh|ksh|fish|python3?|perl|ruby|node)', "download-and-execute"),
    (r'>\s*/dev/(sd[a-z]|nvme\d|hd[a-z])', "block device write"),
    (r'\bsudo\s+rm\s+(-r|-R|--recursive)', "privileged recursive delete"),
    (r'\brm\s+-[a-zA-Z]*r[a-zA-Z]*\s+(/($|\s)|/home\b|/usr\b|/etc\b|/var\b|~\s*(/|$))', "recursive delete of system path"),
]

# Patterns that REQUIRE USER PERMISSION — potentially destructive but sometimes legitimate.
_NEEDS_PERMISSION: list[tuple[str, str]] = [
    (r'\brm\s', "file deletion"),
    (r'\brmdir\b', "directory removal"),
    (r'\bgit\s+push\b', "git push to remote"),
    (r'\bgit\s+reset\s+--hard\b', "destructive git reset"),
    (r'\bgit\s+clean\s+.*-f\b', "git clean (removes untracked files)"),
    (r'\bgit\s+remote\s+add\b', "adding git remote"),
    (r'\bcurl\b', "network request (use web_fetch tool instead)"),
    (r'\bwget\b', "network download"),
    (r'\bpip3?\s+install\b', "package install"),
    (r'\bnpm\s+install\b', "package install"),
    (r'\bcargo\s+install\b', "package install"),
    (r'\bbrew\s+install\b', "package install"),
    (r'\bchmod\b', "file permission change"),
    (r'\bchown\b', "file ownership change"),
    (r'\bsudo\b', "privilege escalation"),
    (r'\bkill\b', "process termination"),
    (r'\bpkill\b', "process termination"),
]


def classify(command: str) -> tuple[str, str]:
    """Return ('blocked' | 'needs_permission' | 'safe', reason)."""
    for pattern, reason in _HARD_BLOCKED:
        if re.search(pattern, command, re.IGNORECASE):
            return "blocked", reason
    for pattern, reason in _NEEDS_PERMISSION:
        if re.search(pattern, command, re.IGNORECASE):
            return "needs_permission", reason
    return "safe", ""
