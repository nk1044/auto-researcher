"""Structured event types for the observability event stream."""

from __future__ import annotations

import asyncio
import datetime
import json
import logging
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Callable, Coroutine, Optional

logger = logging.getLogger(__name__)


class EventType(str, Enum):
    HYPOTHESIS_FORMED = "hypothesis_formed"
    DUP_REJECTED = "dup_rejected"
    DECOMPOSED = "decomposed"
    MODEL_ROUTED = "model_routed"
    SUBAGENT_SPAWNED = "subagent_spawned"
    SUBAGENT_PROGRESS = "subagent_progress"
    SUBAGENT_DONE = "subagent_done"
    REVIEW_INTEGRATE = "review_integrate"
    TEST_SCORED = "test_scored"
    MEMORY_RECORDED = "memory_recorded"
    SAVED = "saved"
    PAUSED = "paused"
    RESUMED = "resumed"
    SHUTDOWN = "shutdown"
    ERROR = "error"
    LOOP_STARTED = "loop_started"
    REWARD_HACK_REJECTED = "reward_hack_rejected"
    FIX_ATTEMPT = "fix_attempt"
    SUBAGENT_STEP = "subagent_step"
    PERMISSION_REQUEST = "permission_request"
    PERMISSION_GRANTED = "permission_granted"
    PERMISSION_DENIED = "permission_denied"
    EXPLORATION_STARTED = "exploration_started"
    EXPLORATION_DONE = "exploration_done"
    INNER_LOOP_CONTINUE = "inner_loop_continue"
    INNER_LOOP_EXIT = "inner_loop_exit"


@dataclass
class Event:
    type: EventType
    data: dict[str, Any]
    ts: str = field(default_factory=lambda: datetime.datetime.utcnow().isoformat())
    iteration: Optional[int] = None

    def to_json(self) -> str:
        return json.dumps(
            {
                "type": self.type.value,
                "data": self.data,
                "ts": self.ts,
                "iteration": self.iteration,
            }
        )


# Subscriber type: async callable that receives an Event
Subscriber = Callable[[Event], Coroutine[Any, Any, None]]


class EventBus:
    """Simple async pub/sub bus. Thread-safe via asyncio queue per subscriber."""

    def __init__(self) -> None:
        self._subscribers: list[asyncio.Queue[Event]] = []
        self._lock = asyncio.Lock()

    async def subscribe(self) -> asyncio.Queue[Event]:
        """Return a queue that receives all future events."""
        q: asyncio.Queue[Event] = asyncio.Queue(maxsize=512)
        async with self._lock:
            self._subscribers.append(q)
        return q

    async def unsubscribe(self, q: asyncio.Queue[Event]) -> None:
        async with self._lock:
            try:
                self._subscribers.remove(q)
            except ValueError:
                pass

    async def publish(self, event: Event) -> None:
        async with self._lock:
            dead: list[asyncio.Queue[Event]] = []
            for q in self._subscribers:
                try:
                    q.put_nowait(event)
                except asyncio.QueueFull:
                    # Slow subscriber — drop oldest event to make room
                    try:
                        q.get_nowait()
                        q.put_nowait(event)
                    except Exception:
                        dead.append(q)
            for d in dead:
                self._subscribers.remove(d)

    def emit(self, event: Event) -> None:
        """Fire-and-forget publish (schedules a coroutine on the running loop)."""
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(self.publish(event))
        except RuntimeError:
            logger.warning("No running event loop; event dropped: %s", event.type)


# Module-level singleton
bus = EventBus()


def emit(
    event_type: EventType,
    data: dict[str, Any],
    iteration: Optional[int] = None,
) -> None:
    """Convenience wrapper for synchronous callers."""
    bus.emit(Event(type=event_type, data=data, iteration=iteration))


async def aemit(
    event_type: EventType,
    data: dict[str, Any],
    iteration: Optional[int] = None,
) -> None:
    """Async emit."""
    await bus.publish(Event(type=event_type, data=data, iteration=iteration))
