# Auto-Researcher

An autonomous AI agent that continuously improves a target software repository. It runs an infinite loop — forming a hypothesis, executing it across parallel subagents, testing the result, and committing improvements to GitHub — without human intervention.

## How It Works

```
form hypothesis
      │
      ▼
anti-repetition gate ──(duplicate failure?)──► reform with novelty
      │
      ▼
decompose into subtasks
      │
      ▼
route each subtask to a specialist model
      │
      ▼
dispatch subagents in parallel ◄── each works in its own git worktree
      │
      ▼
review + integrate diffs
      │
      ▼
run test oracle → score
      │
  score > baseline?
      │ yes
      ▼
validate diff (no test-file tampering)
      │
      ▼
commit + push to GitHub
      │
      ▼
record to memory → repeat
```

The loop only exits when you click **Stop** in the dashboard.

## Features

- **Fully autonomous** — no human in the loop after you click Start
- **Parallel subagents** — one hypothesis, multiple workers, map-reduce integration
- **Skill-based model routing** — sends math tasks to a math model, code tasks to a code model
- **Persistent 3-layer memory** — episodic log, semantic vector search, and resume state
- **Anti-repetition gate** — cosine similarity check blocks re-trying failed approaches
- **Reward-hacking safeguard** — diffs touching test files are rejected before saving
- **Live dashboard** — FastAPI + WebSocket, no build step required
- **Resume after kill** — state is persisted after every iteration

## Quick Start

See [setup.md](setup.md) for installation, configuration, and first-run instructions.

**Once installed:**

1. Set `target_repo` in `config.yaml` to the absolute path of the repo you want to improve
2. Edit `user_tools/test.py` to implement your scoring function (returns a `0.0–1.0` float)
3. Run `uv run python main.py` and open `http://localhost:8000`
4. Click **Start** — the agent runs until you click **Stop**

## Repository Layout

```
auto-researcher/
├── main.py                  — entry point: loads config, starts server
├── config.yaml              — all configuration
├── coordinator/
│   ├── coordinator.py       — infinite loop, hypothesis formation
│   ├── context.py           — context assembly for coordinator calls
│   ├── decomposer.py        — decompose hypothesis into subtasks
│   └── integrator.py        — merge parallel subagent diffs
├── subagent/
│   ├── subagent.py          — ReAct executor (runs in a git worktree)
│   └── context.py           — bounded context assembly per step
├── memory/
│   ├── episodic.py          — SQLite append-only iteration log
│   ├── semantic.py          — LanceDB vector store (RAG + dup detection)
│   └── state.py             — SQLite single-row resume state
├── models/
│   ├── registry.py          — parses config, validates Ollama at startup
│   ├── router.py            — deterministic skill-based model routing
│   └── client.py            — async Ollama HTTP client
├── tools/
│   ├── decorator.py         — @tool decorator and schema generation
│   ├── runtime.py           — sandboxed subprocess execution
│   ├── validator.py         — reward-hacking diff guard
│   └── save_tool.py         — git commit + push to GitHub
├── server/
│   ├── app.py               — FastAPI REST + WebSocket
│   └── events.py            — EventType enum and async EventBus
├── shared/
│   └── types.py             — all shared dataclasses and enums
├── user_tools/              — drop your custom tools here
│   ├── test.py              — sample test oracle (pytest pass-rate)
│   └── sample_action.py     — sample action tool (shell command)
├── dashboard/
│   └── index.html           — live monitoring dashboard
└── tests/                   — unit tests
```

## Extending the Agent

### Add a custom action tool

Create a `.py` file in `user_tools/`. It is auto-discovered at startup:

```python
# user_tools/my_linter.py
from tools.decorator import tool

@tool(name="run_linter", description="Run ruff on workspace", kind="action")
def run_linter(workspace: str) -> dict:
    import subprocess
    r = subprocess.run(["ruff", "check", workspace], capture_output=True, text=True)
    return {"errors": r.stdout.count("\n"), "output": r.stdout[:500]}
```

### Add a specialist model

```yaml
# config.yaml
models:
  workers:
    - name: "sqlcoder:7b"
      skills: ["sql", "database", "query"]
      options: { temperature: 0.2 }
```

Then `ollama pull sqlcoder:7b` and restart. The coordinator will route SQL subtasks to it automatically.

### Replace the test oracle

Edit `user_tools/test.py`. Return `{"score": float, "remark": str}`. The agent maximizes the score.

## Dashboard

The dashboard at `http://localhost:8000` shows:

- Current hypothesis being tested
- Active subagents with their assigned models and status
- Live event log (hypothesis formed, tests scored, improvements saved)
- Current baseline score and iteration count
- Start / Stop / Pause / Resume controls

## Memory and Resume

All state is persisted under `data_dir` (default `./data`). If the process is killed, restart with `uv run python main.py` and click **Start** — the agent resumes from the last completed iteration and saved baseline score.

To start completely fresh: `rm -rf ./data`

## License

MIT
