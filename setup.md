# Setup Guide

## Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) (Python package manager)
- [Ollama](https://ollama.com) running locally (`http://localhost:11434`)
- Git with push access to your target repository's remote

## 1. Clone and install

```bash
git clone <this-repo>
cd auto-researcher
uv sync
```

`uv sync` creates a `.venv` and installs all dependencies from `pyproject.toml` in one step.

## 2. Pull required Ollama models

```bash
ollama pull qwen2.5:14b-instruct    # coordinator model
ollama pull qwen2.5-coder:7b        # default worker / code specialist
ollama pull nomic-embed-text        # embeddings (required)
```

Optional specialist workers (only needed if you enable them in `config.yaml`):

```bash
ollama pull deepseek-r1:8b          # reasoning specialist
ollama pull mathstral:7b            # math specialist
```

You can substitute any Ollama model — just make sure every model referenced in `config.yaml` is pulled before starting.

## 3. Configure config.yaml

`config.yaml` drives all behaviour. Open it and work through each section:

### Target repository (required)

```yaml
target_repo: "/absolute/path/to/your/repo"
```

This is the only required change. Set it to the absolute path of the git repository the agent will improve.

```yaml
worktree_root: "/tmp/auto-researcher/worktrees"   # where isolated worktrees are created
github_remote: "origin"                            # git remote used when saving improvements
github_branch_prefix: "auto-researcher"            # saved branches: auto-researcher/0001, etc.
```

### Ollama connection

```yaml
ollama_host: "http://localhost:11434"
```

Change this if Ollama is running on a different host or port.

### Models

```yaml
models:
  coordinator: "qwen2.5:14b-instruct"   # drives hypothesis formation, decomposition, integration
  default:     "qwen2.5-coder:7b"        # fallback for subtasks with no skill match
  embed:       "nomic-embed-text"         # embedding model — never routed to subtasks
  workers:
    - name: "mathstral:7b"
      skills: ["math", "proof", "numeric"]
      options:
        temperature: 0.2
    - name: "qwen2.5-coder:7b"
      skills: ["code", "refactor", "debug"]
    - name: "deepseek-r1:8b"
      skills: ["reasoning", "planning", "analysis"]
```

- `coordinator` and `default` are mandatory.
- `embed` must be an embedding model (768-dim vectors); `nomic-embed-text` is the recommended default.
- Each `workers` entry maps a model to a list of skill keywords. When the coordinator decomposes a hypothesis into subtasks, each subtask carries `required_skills`; the router picks the worker with the most skill overlap. No match → `default` model is used.
- Add `options` (e.g. `temperature`, `num_ctx`) to pass Ollama model parameters.

To add a new specialist:

```yaml
    - name: "sqlcoder:7b"
      skills: ["sql", "database", "query"]
      options:
        temperature: 0.1
```

Then `ollama pull sqlcoder:7b` and restart.

### Loop control

```yaml
max_subagents: 4        # max concurrent subagents spawned per iteration
dup_threshold: 0.97     # cosine similarity above which a hypothesis is rejected as a near-duplicate of a prior failure
novelty_boost: 0.3      # temperature increase applied when the coordinator is forced to generate a novel hypothesis
```

- Raise `max_subagents` for broader parallelism (uses more RAM/GPU).
- Lower `dup_threshold` (e.g. `0.90`) to be more aggressive at blocking similar failed hypotheses. Raise it (e.g. `0.99`) to allow more near-duplicate retries.

### Context budgets

```yaml
context_token_budget: 8192

coordinator_context_ratios:
  task_spec:       0.20   # system prompt + tool schemas
  memory:          0.30   # retrieved past iterations
  files:           0.35   # current file slices from target repo
  rolling_summary: 0.15   # periodically regenerated iteration summary

subagent_context_ratios:
  task_brief:      0.20   # subtask brief + tool schemas
  files:           0.50   # in-scope file slices
  memory:          0.15   # narrowly retrieved memory
  rolling_summary: 0.15   # subagent self-summary
```

Increase `context_token_budget` if your models support longer context and you want the agent to see more of the codebase per turn. Ratios must sum to 1.0 within each group.

### Subagent execution

```yaml
subagent_step_cap: 20    # max ReAct steps a subagent may take before it is terminated
summary_every_n:   5     # regenerate rolling summary every N steps
```

### Sandbox

```yaml
sandbox:
  max_cpu_seconds: 120   # CPU time limit per tool call subprocess
  max_memory_mb:   512   # RSS memory limit per subprocess
  timeout_seconds: 300   # wall-clock timeout per tool call
```

These limits apply to sandboxed subprocess tool calls (e.g. running linters, test oracles).

### Persistence

```yaml
data_dir: "./data"
```

All persistent state lives here: `episodic.db` (iteration log), `state.db` (resume state), `semantic_db/` (LanceDB vector store). To reset the agent completely, delete this directory.

### Server

```yaml
server:
  host: "0.0.0.0"
  port: 8000
```

### Reward-hacking safeguards

```yaml
protected_patterns:
  - "tests/"
  - "test/"
  - "held_out/"
  - "eval/"
  - "benchmark/"
  - "*.test.*"
  - "test_*.py"
  - "*_test.py"
```

Any diff that touches a file matching one of these patterns is rejected before saving. Add patterns for your held-out evaluation data or any files the agent must never modify.

## 4. Provide your test oracle

The agent needs a scoring function that measures improvement. Edit `user_tools/test.py`:

```python
from tools.decorator import tool

@tool(name="run_tests", description="Run tests and return a 0-1 score", kind="test")
def run_tests(workspace: str) -> dict:
    # workspace is the absolute path to the candidate worktree
    # run your evaluation logic here
    return {"score": 0.75, "remark": "60/80 tests passed"}
```

The only contract: return `{"score": float, "remark": str}` where `score` is in `[0.0, 1.0]`. The agent maximizes it. The oracle is opaque — the agent never sees its source.

## 5. Start the server

```bash
# Start server — loop does NOT begin automatically
uv run python main.py

# Start server and immediately begin the improvement loop
uv run python main.py --config config.yaml --autostart
```

Open the dashboard at `http://localhost:8000`.

## 6. Dashboard controls

| Button | Action |
|--------|--------|
| **Start** | Begin the infinite improvement loop |
| **Pause** | Suspend after the current iteration finishes |
| **Resume** | Unpause |
| **Stop** | Clean shutdown — drains in-flight work, flushes memory |

## 7. API endpoints

```
POST /start    — start the loop
POST /stop     — request shutdown
POST /pause    — pause (blocks until current iteration finishes)
POST /resume   — resume
GET  /state    — current state as JSON
WS   /events   — live structured event stream
```

## 8. Running tests

```bash
PYTHONPATH=. uv run pytest tests/ -v
```

Tests do not require a running Ollama instance or GitHub connection — all network-dependent behaviour is mocked. `PYTHONPATH=.` is required so that `shared`, `tools`, `memory`, etc. are importable as top-level packages.

## 9. Resume after restart

If the process is killed, restart with the same command. The coordinator reads from `data/` to restore:

- iteration counter
- current baseline score
- working git commit

The loop continues from where it left off.

## 10. Adding custom tools

Drop any `.py` file into `user_tools/`. It is auto-discovered at startup. Use `@tool` for explicit control:

```python
from tools.decorator import tool

@tool(name="lint", description="Run ruff on workspace", kind="action")
def run_linter(workspace: str) -> dict:
    import subprocess
    r = subprocess.run(["ruff", "check", workspace], capture_output=True, text=True)
    return {"errors": r.stdout.count("\n"), "output": r.stdout[:500]}
```

Functions without `@tool` are also auto-registered with schemas inferred from type hints.
