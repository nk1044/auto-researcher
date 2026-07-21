"""FastAPI server: REST control endpoints + WebSocket event stream."""

from __future__ import annotations

import asyncio
import json
import logging
import os
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse

from server.events import Event, EventType, aemit, bus

logger = logging.getLogger(__name__)

app = FastAPI(title="Auto-Researcher", version="1.0.0")

# ── Global coordinator reference (set by main.py after startup) ──────────────
_coordinator: Any = None
_coordinator_task: Optional[asyncio.Task] = None


def set_coordinator(coordinator: Any) -> None:
    global _coordinator
    _coordinator = coordinator


# ── Static dashboard ──────────────────────────────────────────────────────────
_dashboard_path = Path(__file__).parent.parent / "dashboard" / "index.html"


@app.get("/", response_class=HTMLResponse)
async def dashboard() -> HTMLResponse:
    if _dashboard_path.exists():
        return HTMLResponse(_dashboard_path.read_text())
    return HTMLResponse("<h1>Dashboard not found</h1>", status_code=404)


# ── Control endpoints ─────────────────────────────────────────────────────────

@app.post("/start")
async def start() -> JSONResponse:
    global _coordinator_task, _coordinator
    if _coordinator is None:
        return JSONResponse({"error": "coordinator not initialised"}, status_code=503)
    if _coordinator_task and not _coordinator_task.done():
        return JSONResponse({"status": "already running"})
    _coordinator_task = asyncio.create_task(_coordinator.run())
    return JSONResponse({"status": "started"})


@app.post("/stop")
async def stop() -> JSONResponse:
    if _coordinator is None:
        return JSONResponse({"error": "coordinator not initialised"}, status_code=503)
    _coordinator.stop()
    return JSONResponse({"status": "stop_requested"})


@app.post("/pause")
async def pause() -> JSONResponse:
    if _coordinator is None:
        return JSONResponse({"error": "coordinator not initialised"}, status_code=503)
    _coordinator.pause()
    await aemit(EventType.PAUSED, {})
    return JSONResponse({"status": "paused"})


@app.post("/resume")
async def resume() -> JSONResponse:
    if _coordinator is None:
        return JSONResponse({"error": "coordinator not initialised"}, status_code=503)
    _coordinator.resume()
    await aemit(EventType.RESUMED, {})
    return JSONResponse({"status": "resumed"})


@app.get("/state")
async def get_state() -> JSONResponse:
    if _coordinator is None:
        return JSONResponse({"error": "coordinator not initialised"}, status_code=503)
    c = _coordinator
    running = _coordinator_task is not None and not _coordinator_task.done()
    return JSONResponse(
        {
            "iteration": c.state.iteration,
            "baseline_score": c.baseline,
            "working_commit": c.state.working_commit[:8] if c.state.working_commit else "",
            "current_hypothesis": c.current_hypothesis,
            "running": running,
            "paused": not c.pause_gate.is_set(),
        }
    )


# ── WebSocket event stream ────────────────────────────────────────────────────

@app.websocket("/events")
async def websocket_events(ws: WebSocket) -> None:
    await ws.accept()
    q = await bus.subscribe()
    logger.debug("WebSocket client connected")
    try:
        while True:
            try:
                event: Event = await asyncio.wait_for(q.get(), timeout=15.0)
                await ws.send_text(event.to_json())
            except asyncio.TimeoutError:
                # Send keepalive ping
                try:
                    await ws.send_text('{"type":"ping"}')
                except Exception:
                    break
    except WebSocketDisconnect:
        logger.debug("WebSocket client disconnected")
    finally:
        await bus.unsubscribe(q)
