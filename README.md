# Auto-Researcher

An autonomous AI agent that continuously improves a target software repository. It runs an infinite loop — forming a hypothesis, executing it across parallel subagents, testing the result, and committing improvements to GitHub — without human intervention.

## How It Works

The agent runs a single feedback loop forever:

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

## Prerequisites

- Python 3.11+
- [Ollama](https://ollama.com) running locally (or on a remote host)
- Git (with push access to your target repo's remote)
- The following models pulled in Ollama:

```bash
ollama pull qwen2.5:14b-instruct
ollama pull qwen2.5-coder:7b
ollama pull nomic-embed-text
```

Optional specialist workers (add more in `config.yaml`):

```bash
ollama pull mathstral:7b
ollama pull deepseek-r1:8b
```

## Installation

```bash
git clone <this-repo>
cd auto-researcher
pip install -r requirements.txt
```

## Quick Start

**1. Edit `config.yaml`**

Set `target_repo` to the absolute path of the repository you want to improve:

```yaml
target_repo: "/absolute/path/to/your/repo"
```

That is the only required change. Everything else has sensible defaults.

**2. Add your test oracle**

Edit `user_tools/test.py` to implement your scoring function. The default runs `pytest` and returns a pass-rate score:

```python
@tool(name="run_tests", description="Run tests and return a 0-1 score", kind="test")
def run_tests(workspace: str) -> dict:
    # your evaluation logic here
    return {"score": 0.75, "remark": "60/80 tests passed"}
```

The score must be a float in `[0.0, 1.0]`. The agent maximizes it.

**3. Start the server**

```bash
python main.py
```

**4. Open the dashboard**

Go to `http://localhost:8000` and click **Start**.

The agent begins immediately and runs until you click **Stop**.

## Configuration

All behaviour is driven by `config.yaml`. Key settings:

| Key | Default | Description |
|-----|---------|-------------|
| `target_repo` | *(required)* | Absolute path to the repo to improve |
| `ollama_host` | `http://localhost:11434` | Ollama server URL |
| `models.coordinator` | `qwen2.5:14b-instruct` | Model for reasoning (hypothesis, decompose, integrate) |
| `models.default` | `qwen2.5-coder:7b` | Fallback model for subagents |
| `models.embed` | `nomic-embed-text` | Embedding model (768-dim) |
| `max_subagents` | `4` | Max concurrent subagents per iteration |
| `subagent_step_cap` | `20` | Max ReAct steps per subagent |
| `context_token_budget` | `8192` | Token budget for context assembly |
| `dup_threshold` | `0.92` | Cosine similarity cutoff for duplicate-failure detection |
| `data_dir` | `./data` | Where episodic, semantic, and state storage lives |
| `github_branch_prefix` | `auto-researcher` | Branch prefix for saved improvements |
| `protected_patterns` | `tests/`, `test_*.py`, … | Files the agent is never allowed to modify |

See [docs/08_configuration.md](docs/08_configuration.md) for a full reference of every key.

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
│   ├── __init__.py          — Memory façade
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
├── tests/                   — unit tests
└── docs/                    — contributor documentation (chapters 0–9)
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

Edit `user_tools/test.py`. Return `{"score": float, "remark": str}`. The agent maximizes the score. See [docs/09_contributing.md](docs/09_contributing.md) for multi-metric scoring examples.

## Dashboard

The dashboard at `http://localhost:8000` shows:

- Current hypothesis being tested
- Active subagents with their assigned models and status
- Live event log (hypothesis formed, tests scored, improvements saved)
- Current baseline score and iteration count
- Start / Stop / Pause / Resume controls

## Memory and Resume

All state is persisted under `data_dir` (default `./data`). If the process is killed, restart with `python main.py` and click **Start** — the agent resumes from the last completed iteration and the last saved baseline score.

To start completely fresh: `rm -rf ./data`

## Documentation

The `docs/` folder contains a detailed chapter-by-chapter walkthrough for contributors:

| Chapter | Topic |
|---------|-------|
| [00 — Overview](docs/00_overview.md) | System design, loop diagram, reading order |
| [01 — Shared Types](docs/01_shared_types.md) | All dataclasses and enums |
| [02 — Memory](docs/02_memory.md) | Three storage layers, cosine similarity, resume |
| [03 — Models](docs/03_models.md) | Registry, routing algorithm, Ollama client |
| [04 — Tools](docs/04_tools.md) | @tool decorator, sandbox, validator, save |
| [05 — Subagent](docs/05_subagent.md) | ReAct loop, worktree isolation, context assembly |
| [06 — Coordinator](docs/06_coordinator.md) | Infinite loop, decomposition, integration |
| [07 — Server & Dashboard](docs/07_server_dashboard.md) | REST API, WebSocket events, dashboard JS |
| [08 — Configuration](docs/08_configuration.md) | Every config key explained |
| [09 — Contributing](docs/09_contributing.md) | How to extend the system |

## License

MIT
