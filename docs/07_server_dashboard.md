# Chapter 7 — Server and Dashboard (`server/`)

## Overview

The server module provides two things:

1. **A REST API** to control the agent (start, stop, pause, resume, inspect state).
2. **A live WebSocket event stream** so a browser can watch the agent work in real time.
3. **A static dashboard** — plain HTML + JavaScript that consumes both.

The server is not optional. It is how you interact with the running agent. The coordinator runs inside the server process, not as a separate daemon.

---

## Files

```
server/
├── app.py         — FastAPI application (REST endpoints + WebSocket + static file)
├── events.py      — EventType enum, Event dataclass, EventBus class
dashboard/
└── index.html     — Single-file dashboard (HTML + CSS + JavaScript)
```

---

## Events System (`server/events.py`)

### EventType

```python
class EventType(str, Enum):
    LOOP_STARTED        = "loop_started"
    HYPOTHESIS_FORMED   = "hypothesis_formed"
    DUP_REJECTED        = "dup_rejected"
    DECOMPOSED          = "decomposed"
    MODEL_ROUTED        = "model_routed"
    SUBAGENT_SPAWNED    = "subagent_spawned"
    SUBAGENT_PROGRESS   = "subagent_progress"
    SUBAGENT_DONE       = "subagent_done"
    REVIEW_INTEGRATE    = "review_integrate"
    TEST_SCORED         = "test_scored"
    MEMORY_RECORDED     = "memory_recorded"
    SAVED               = "saved"
    PAUSED              = "paused"
    RESUMED             = "resumed"
    SHUTDOWN            = "shutdown"
    ERROR               = "error"
    REWARD_HACK_REJECTED = "reward_hack_rejected"
```

These event types map to the major steps in `_run_one_iteration()`. Every significant state change emits an event. The dashboard subscribes and reacts to these — when it sees `TEST_SCORED`, it updates the score display; when it sees `SUBAGENT_SPAWNED`, it adds a row to the subagents table.

### Event Dataclass

```python
@dataclass
class Event:
    type:      EventType
    data:      dict        # event-specific payload
    iteration: int | None  # which iteration this belongs to
    ts:        str         # ISO 8601 timestamp (UTC)
```

Events are serialized to JSON for the WebSocket stream. Every event carries the iteration number so the dashboard can group them.

### EventBus

```python
class EventBus:
    _subscribers: list[asyncio.Queue]

    async def subscribe(self) -> asyncio.Queue[Event]:
        # Creates a new Queue for this subscriber
        # Adds it to _subscribers list
        # Returns the queue

    async def unsubscribe(self, q: asyncio.Queue) -> None:
        # Removes the queue from _subscribers

    async def publish(self, event: Event) -> None:
        # Puts event into every subscriber's queue
        # Missing subscribers don't block the event (put_nowait)

    def emit(self, event: Event) -> None:
        # Fire-and-forget: creates a task via loop.create_task
        # Safe to call from sync code
```

The `emit()` method is the sync-safe way to fire events from inside the coordinator. It schedules `publish()` as a coroutine without awaiting it — so the coordinator never blocks waiting for dashboard clients.

`aemit()` is the async convenience function used throughout the coordinator:

```python
async def aemit(event_type: EventType, data: dict, iteration: int | None = None) -> None:
    event = Event(type=event_type, data=data, iteration=iteration, ts=now())
    await bus.publish(event)
```

`bus` is a module-level singleton — it is created once when `server.events` is imported and shared by everything. There is no dependency injection here; everyone imports `bus` or `aemit` directly.

---

## FastAPI Application (`server/app.py`)

### REST Endpoints

```python
@app.post("/start")
async def start() -> dict:
    # Creates and starts the coordinator in a background task
    # Returns {"status": "started"} or {"status": "already_running"}

@app.post("/stop")
async def stop() -> dict:
    # Calls coordinator.stop()
    # Returns {"status": "stopping"}

@app.post("/pause")
async def pause() -> dict:
    # Calls coordinator.pause()
    # Returns {"status": "paused"}

@app.post("/resume")
async def resume() -> dict:
    # Calls coordinator.resume()
    # Returns {"status": "resumed"}

@app.get("/state")
async def get_state() -> dict:
    # Returns current agent state:
    # {
    #   "running": bool,
    #   "paused": bool,
    #   "iteration": int,
    #   "baseline_score": float,
    #   "working_commit": str,
    #   "current_hypothesis": str | null
    # }
```

These are thin wrappers over the coordinator's control methods. The coordinator instance is stored in a module-level variable `_coordinator` and set by `set_coordinator()` (called from `main.py`).

### State Consistency

The `/state` endpoint is safe to call at any time, even while an iteration is running. It reads from `coordinator.state` which is an `AgentState` dataclass updated atomically at the end of each iteration. Reading it mid-iteration may show the previous iteration's state — this is acceptable for a monitoring dashboard.

### WebSocket: `/events`

```python
@app.websocket("/events")
async def events_ws(ws: WebSocket) -> None:
    await ws.accept()
    q = await bus.subscribe()
    try:
        while True:
            event = await q.get()
            await ws.send_text(event.type + ":" + json.dumps({
                "data": event.data,
                "iteration": event.iteration,
                "ts": event.ts,
            }))
    except WebSocketDisconnect:
        pass
    finally:
        await bus.unsubscribe(q)
```

The wire format is `"event_type:json_payload"`. The dashboard splits on the first `:` to get the type, then parses the rest as JSON. This simple format avoids an extra JSON nesting layer.

When the client disconnects (`WebSocketDisconnect`), the subscription is cleaned up in the `finally` block. There is no leak.

### Static Dashboard

```python
@app.get("/", response_class=HTMLResponse)
async def dashboard() -> str:
    path = Path(__file__).parent.parent / "dashboard" / "index.html"
    return path.read_text()
```

The dashboard is served from a file read, not as a static mount. This is intentional — it avoids a dependency on `StaticFiles` and keeps the setup simple.

---

## Dashboard (`dashboard/index.html`)

The dashboard is a single HTML file. No build step. No dependencies. It loads in any browser.

### Layout

```
┌─────────────────────────────────────────────────────────────────┐
│  Auto-Researcher Dashboard                                       │
│  [Start] [Stop] [Pause] [Resume]                                 │
├─────────────────────────────────────────────────────────────────┤
│  Status: RUNNING  |  Iteration: 42  |  Score: 0.823  |  Commit: abc123
├─────────────────────────────────────────────────────────────────┤
│  Current Hypothesis:                                             │
│  "Refactor the tokenizer to handle edge cases in..."            │
├────────────────────┬────────────────────────────────────────────┤
│  Subagents         │  Event Log                                  │
│  ID     Model  Stat│  [test_scored] score=0.823, remark=60/80   │
│  sa-001 qwen7b DONE│  [subagent_done] sa-001: SUCCESS            │
│  sa-002 qwen7b RUN │  [hypothesis_formed] Refactor the...        │
└────────────────────┴────────────────────────────────────────────┘
```

### JavaScript Architecture

The dashboard has three sections of JavaScript:

**1. Control buttons**

Each button makes a `fetch()` call to the corresponding REST endpoint:

```javascript
document.getElementById("startBtn").onclick = () =>
    fetch("/start", {method: "POST"}).then(r => r.json()).then(updateStatus);
```

**2. State polling**

Every 5 seconds, the dashboard calls `GET /state` and updates the status bar:

```javascript
setInterval(() => {
    fetch("/state").then(r => r.json()).then(state => {
        document.getElementById("iteration").textContent = state.iteration;
        document.getElementById("score").textContent = state.baseline_score.toFixed(3);
        ...
    });
}, 5000);
```

**3. WebSocket event listener**

On page load, a WebSocket connection is opened to `/events`:

```javascript
const ws = new WebSocket(`ws://${location.host}/events`);

ws.onmessage = (msg) => {
    const colon = msg.data.indexOf(":");
    const eventType = msg.data.slice(0, colon);
    const payload = JSON.parse(msg.data.slice(colon + 1));
    handleEvent(eventType, payload);
};
```

The `handleEvent()` function dispatches on `eventType` and updates the appropriate DOM element:

```javascript
function handleEvent(type, payload) {
    addToEventLog(type, payload);  // always appends to the log

    switch (type) {
        case "hypothesis_formed":
            document.getElementById("hypothesis").textContent = payload.data.hypothesis;
            break;
        case "subagent_spawned":
            addSubagentRow(payload.data);
            break;
        case "subagent_done":
            updateSubagentRow(payload.data.subtask_id, payload.data.status);
            break;
        case "test_scored":
            document.getElementById("score").textContent = payload.data.score.toFixed(3);
            break;
        case "saved":
            addSavedBadge(payload.data.branch, payload.data.score);
            break;
        case "reward_hack_rejected":
            highlightError(payload.data.reason);
            break;
    }
}
```

### Reconnection

The WebSocket connection may drop (network glitch, server restart). The dashboard handles this with exponential backoff reconnection:

```javascript
function connectWS() {
    const ws = new WebSocket(`ws://${location.host}/events`);
    ws.onclose = () => setTimeout(connectWS, 2000);  // reconnect after 2s
    ws.onmessage = ...;
}
connectWS();
```

### Event Log

The event log is a scrollable `<div>` with a maximum height and `overflow-y: auto`. New events are prepended (most recent first):

```javascript
function addToEventLog(type, payload) {
    const log = document.getElementById("eventLog");
    const row = document.createElement("div");
    row.className = `event-row ${type}`;
    row.textContent = `[${payload.ts}] [${type}] ${JSON.stringify(payload.data)}`;
    log.prepend(row);
    if (log.children.length > 200) log.lastChild.remove();  // cap at 200 events
}
```

The `type` class is added to each row so CSS can color-code events:

```css
.event-row.test_scored    { color: #4caf50; }
.event-row.error          { color: #f44336; }
.event-row.reward_hack_rejected { color: #ff9800; background: #fff3e0; }
.event-row.saved          { color: #2196f3; font-weight: bold; }
```

---

## Starting the Server

```bash
python main.py --config config.yaml
```

Under the hood, `main.py` calls:

```python
import uvicorn
uvicorn.run(server_app.app, host="0.0.0.0", port=8000, log_level="info")
```

The agent does **not** auto-start. After the server is up, open `http://localhost:8000` and click **Start** to begin the loop. This gives you a chance to verify the config and check the dashboard layout before the first iteration runs.

---

## For Contributors

**Adding a new REST endpoint:**
Add it to `server/app.py`. The `_coordinator` module-level variable is available to all endpoint handlers. Do not import from `coordinator` inside route handlers — use `_coordinator` from `app.py`.

**Adding a new event type:**
1. Add the value to `EventType` in `server/events.py`.
2. Call `await aemit(EventType.YOUR_EVENT, {...})` in the coordinator.
3. Add a `case "your_event":` branch in the dashboard's `handleEvent()`.

**Replacing the dashboard:**
Replace `dashboard/index.html` with any frontend. It just needs to connect to `/events` (WebSocket) and call the REST endpoints. The API is stable — the HTML file is the only thing that needs to change.

**Embedding metrics in events:**
All event payloads are arbitrary dicts. If you want to track token usage, model latency, or subagent step counts in the dashboard, add those fields to the event payload in the coordinator and read them in `handleEvent()`. The event bus does not restrict payload shapes.
