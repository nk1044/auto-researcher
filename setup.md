# Setup and Run Guide

## Prerequisites

- Python 3.11+
- [Ollama](https://ollama.com) running locally (`http://localhost:11434`)
- Git (with a configured remote if you want GitHub save to work)
- The target repository you want to improve must be a git repo

## 1. Install dependencies

```bash
pip install -r requirements.txt
```

## 2. Pull required Ollama models

The default config uses these models — pull whichever you want to use:

```bash
ollama pull qwen2.5:14b-instruct    # coordinator
ollama pull qwen2.5-coder:7b        # default worker / code specialist
ollama pull nomic-embed-text        # embeddings (required)
ollama pull deepseek-r1:8b          # reasoning specialist (optional)
ollama pull mathstral:7b            # math specialist (optional)
```

You can substitute any Ollama model in `config.yaml` — just make sure all referenced
models are pulled before starting.

## 3. Configure

Edit `config.yaml`:

```yaml
target_repo: "/absolute/path/to/your/repo"   # REQUIRED
```

Everything else has sensible defaults. Key options:

| Key | Default | Description |
|-----|---------|-------------|
| `models.coordinator` | `qwen2.5:14b-instruct` | Model driving hypothesis/decompose/integrate |
| `models.default` | `qwen2.5-coder:7b` | Fallback for unmatched subtasks |
| `models.embed` | `nomic-embed-text` | Embeddings model (never routed) |
| `max_subagents` | `4` | Max parallel subagents per iteration |
| `dup_threshold` | `0.92` | Cosine similarity threshold for duplicate rejection |
| `context_token_budget` | `8192` | Hard token cap per agent turn |
| `github_remote` | `origin` | Git remote for saving improvements |
| `github_branch_prefix` | `auto-researcher` | Branch name prefix: `auto-researcher/0001` |

## 4. Provide your test tool

The agent needs an opaque test oracle. The sample in `user_tools/test.py` runs pytest
and returns a pass-rate score. Replace or extend it with your own evaluation logic.

The only contract:
```python
def run_tests(workspace: str) -> dict:
    ...
    return {"score": 0.75, "remark": "60/80 tests passed"}
```

`score` must be a float in `[0, 1]` (or any monotonically comparable float). The agent
improves toward higher scores.

## 5. Start the server

```bash
# Start server only (loop does NOT start automatically — use the dashboard STOP/Start button)
python main.py --config config.yaml

# Start server AND immediately begin the improvement loop
python main.py --config config.yaml --autostart
```

Open the dashboard at `http://localhost:8000`.

## 6. Dashboard controls

| Button | Action |
|--------|--------|
| **Start** | Begin the infinite improvement loop |
| **Pause** | Suspend after current iteration finishes |
| **Resume** | Unpause |
| **STOP** | Request clean shutdown (drains in-flight work, flushes memory) |

## 7. API endpoints

```
POST /start    — start the loop
POST /stop     — request shutdown
POST /pause    — pause (blocks until current iteration finishes)
POST /resume   — resume
GET  /state    — current state as JSON
WS   /events   — live structured event stream (see server/events.py for types)
```

## 8. Running tests

```bash
# Standard (from project root):
PYTHONPATH=. python3 -c "import pytest,sys; sys.exit(pytest.main(['tests/', '-v'], plugins=[]))"

# Or if your environment has no conflicting pytest plugins:
PYTHONPATH=. pytest tests/ -v
```

Tests do NOT require a running Ollama or GitHub connection. All network-dependent
behaviour is mocked. The `PYTHONPATH=.` is required so that `shared`, `tools`,
`memory`, etc. are importable as top-level packages.

## 9. Resume after restart

If the process is killed (`kill -9`, crash, etc.), restart with the same command.
The coordinator automatically reads from `data/` to restore:
- iteration counter
- current baseline score  
- working git commit

The loop continues from where it left off.

## 10. Adding custom tools

Drop any `.py` file into `user_tools/`. Functions that return a plain value (dict,
str, int, etc.) are auto-discovered and registered. Use the `@tool` decorator if you
want explicit name/description/kind control:

```python
from tools.decorator import tool

@tool(name="lint", description="Run ruff on workspace", kind="action")
def run_linter(workspace: str) -> dict:
    ...
    return {"errors": 0, "warnings": 3}
```

Functions without `@tool` are also auto-registered with auto-generated schemas based
on type hints.

## Project layout

```
auto-researcher/
├── main.py                  # entry point
├── config.yaml              # all configuration
├── requirements.txt
├── coordinator/             # infinite loop, hypothesis, decompose, integrate
├── subagent/                # isolated ReAct executor
├── memory/                  # episodic (SQLite) + semantic (LanceDB) + state
├── models/                  # registry, skill router, Ollama client
├── tools/                   # decorator, sandboxed runtime, validator, save
├── server/                  # FastAPI + WebSocket event bus
├── dashboard/               # plain HTML/JS live dashboard
├── user_tools/              # user-supplied test oracle + action tools
├── shared/                  # shared data types
└── tests/                   # pytest test suite
```
