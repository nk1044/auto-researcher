"""Permission request/response manager for dangerous shell commands.

When a subagent wants to run a command that needs user approval, the flow is:
  1. ToolRuntime calls `await request(command, reason, requester_id)`
  2. A PERMISSION_REQUEST event is published to the WebSocket event bus
  3. The dashboard shows a modal asking the user to Approve or Deny
  4. User clicks → browser POSTs to /permission/{req_id}/approve  or /deny
  5. Server calls `resolve(req_id, approved=True/False)`
  6. The asyncio.Event is set, `request()` returns the decision
  7. ToolRuntime either proceeds or returns a denial error to the subagent
"""

from __future__ import annotations

import asyncio
import logging
import uuid

logger = logging.getLogger(__name__)

# req_id -> asyncio.Event (set when user responds)
_pending: dict[str, asyncio.Event] = {}
# req_id -> bool (True = approved)
_decisions: dict[str, bool] = {}

_DEFAULT_TIMEOUT = 120.0   # seconds to wait before auto-denying


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
        "Permission request %s from %s: [%s] %s",
        req_id, requester_id, reason, command[:120],
    )

    try:
        await asyncio.wait_for(event.wait(), timeout=timeout)
        approved = _decisions.get(req_id, False)
    except asyncio.TimeoutError:
        approved = False
        logger.warning("Permission request %s timed out — denied", req_id)

    _pending.pop(req_id, None)
    _decisions.pop(req_id, None)

    from server.events import EventType, aemit
    evt = EventType.PERMISSION_GRANTED if approved else EventType.PERMISSION_DENIED
    await aemit(evt, {"request_id": req_id, "command": command[:120], "requester_id": requester_id})

    return approved


def resolve(req_id: str, approved: bool) -> bool:
    """Called by the server endpoint when the user responds.

    Returns True if the request was found and resolved, False if unknown.
    """
    if req_id not in _pending:
        return False
    _decisions[req_id] = approved
    _pending[req_id].set()
    return True


def pending_ids() -> list[str]:
    return list(_pending.keys())
