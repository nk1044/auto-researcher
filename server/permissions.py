"""Permission request/response manager for dangerous shell commands.

Flow when a subagent wants to run a risky command:
  1. ToolRuntime calls `await request(command, reason, requester_id)`
  2. A PERMISSION_REQUEST event is published to the WebSocket bus
     → Dashboard shows a modal with Approve / Deny buttons (3-minute countdown)
  3. If Discord is configured, a notification is sent IMMEDIATELY
     → User gets pinged even if dashboard is not open
  4. User responds:
       - Via dashboard  → browser POSTs to /permission/{req_id}/approve or /deny
       - Via timeout    → auto-denied after 3 minutes
  5. `request()` returns True (approved) or False (denied)
"""

from __future__ import annotations

import asyncio
import json
import logging
import urllib.request
import urllib.error
import uuid

logger = logging.getLogger(__name__)

_DEFAULT_TIMEOUT = 180.0   # 3 minutes

# Module-level config — set by main.py via configure()
_webhook_url: str = ""
_dashboard_url: str = "http://localhost:8000"

# req_id -> asyncio.Event
_pending: dict[str, asyncio.Event] = {}
# req_id -> bool
_decisions: dict[str, bool] = {}


def configure(webhook_url: str = "", dashboard_url: str = "http://localhost:8000") -> None:
    """Called by main.py at startup to inject runtime config."""
    global _webhook_url, _dashboard_url
    _webhook_url = webhook_url.strip()
    _dashboard_url = dashboard_url.rstrip("/")
    if _webhook_url:
        logger.info("Discord notifications enabled (webhook configured)")
    else:
        logger.info("Discord notifications disabled (no webhook_url in config)")


async def request(
    command: str,
    reason: str,
    requester_id: str,
    timeout: float = _DEFAULT_TIMEOUT,
) -> bool:
    """Publish a permission request and wait for the user's decision.

    Returns True if approved, False if denied or timed out.
    """
    from server.events import EventType, aemit

    req_id = uuid.uuid4().hex[:8]
    event = asyncio.Event()
    _pending[req_id] = event

    await aemit(
        EventType.PERMISSION_REQUEST,
        {
            "request_id": req_id,
            "command": command,
            "reason": reason,
            "requester_id": requester_id,
            "timeout": timeout,
        },
    )
    logger.info(
        "Permission request %s from agent %.8s: [%s] %s",
        req_id, requester_id, reason, command[:120],
    )

    # Fire-and-forget Discord notification immediately
    if _webhook_url:
        asyncio.create_task(
            _notify_discord(req_id, command, reason, requester_id, timeout)
        )

    try:
        await asyncio.wait_for(event.wait(), timeout=timeout)
        approved = _decisions.get(req_id, False)
    except asyncio.TimeoutError:
        approved = False
        logger.warning("Permission request %s timed out (%.0fs) — auto-denied", req_id, timeout)

    _pending.pop(req_id, None)
    _decisions.pop(req_id, None)

    from server.events import EventType, aemit
    evt = EventType.PERMISSION_GRANTED if approved else EventType.PERMISSION_DENIED
    await aemit(evt, {
        "request_id": req_id,
        "command": command[:120],
        "requester_id": requester_id,
        "approved": approved,
    })

    return approved


def resolve(req_id: str, approved: bool) -> bool:
    """Called by the server endpoint when the user responds.

    Returns True if the request was found and resolved, False if unknown/expired.
    """
    if req_id not in _pending:
        return False
    _decisions[req_id] = approved
    _pending[req_id].set()
    return True


def pending_ids() -> list[str]:
    return list(_pending.keys())


# ── Discord notification ────────────────────────────────────────────────────────

async def _notify_discord(
    req_id: str,
    command: str,
    reason: str,
    requester_id: str,
    timeout: float,
) -> None:
    """Send a Discord embed notification about a pending permission request."""
    minutes = int(timeout // 60)
    seconds = int(timeout % 60)
    timeout_str = f"{minutes}m {seconds}s" if minutes else f"{seconds}s"

    embed = {
        "title": "⚠️ Permission Required — Auto-Researcher",
        "description": (
            f"A subagent wants to run a potentially dangerous command "
            f"and is waiting for your approval."
        ),
        "color": 16753920,    # orange
        "fields": [
            {
                "name": "Agent",
                "value": f"`{requester_id[:8]}`",
                "inline": True,
            },
            {
                "name": "Risk",
                "value": reason,
                "inline": True,
            },
            {
                "name": "Command",
                "value": f"```\n{command[:400]}\n```",
                "inline": False,
            },
            {
                "name": "Respond",
                "value": (
                    f"Open the dashboard to **Approve** or **Deny**:\n"
                    f"{_dashboard_url}"
                ),
                "inline": False,
            },
        ],
        "footer": {
            "text": f"Request ID: {req_id} • Auto-denied in {timeout_str} if no response"
        },
    }

    body = json.dumps({"embeds": [embed]}).encode("utf-8")
    req = urllib.request.Request(
        _webhook_url,
        data=body,
        headers={"Content-Type": "application/json", "User-Agent": "auto-researcher/1.0"},
        method="POST",
    )
    try:
        # Run in executor to avoid blocking the event loop
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, lambda: urllib.request.urlopen(req, timeout=10))
        logger.info("Discord notification sent for permission request %s", req_id)
    except Exception as exc:
        logger.warning("Discord notification failed for %s: %s", req_id, exc)
