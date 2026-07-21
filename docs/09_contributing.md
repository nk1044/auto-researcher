# Chapter 9 — Contributing Guide

## Overview

This chapter is a practical guide for extending the system. It covers the most common extension points:
- Adding custom tools for subagents to use.
- Adding or replacing worker models.
- Replacing the test oracle with your own scoring function.
- Extending the memory system.
- Customizing the coordinator's reasoning behavior.

Each section explains what to change, where, and what to watch out for.

---

## Adding a Custom Action Tool

Action tools are what subagents use to interact with the codebase: reading files, running linters, applying patches, querying APIs.

### Step 1: Create the file

```python
# user_tools/my_tool.py
from tools.decorator import tool

@tool(name="run_linter", description="Run ruff linter on workspace and return error count", kind="action")
def run_linter(workspace: str) -> dict:
    import subprocess
    result = subprocess.run(
        ["ruff", "check", workspace, "--output-format", "concise"],
        capture_output=True, text=True, cwd=workspace,
    )
    lines = [l for l in result.stdout.splitlines() if l.strip()]
    return {"errors": len(lines), "output": "\n".join(lines[:20])}
```

### Step 2: Restart the server

`auto_discover("user_tools")` runs at startup. All `.py` files in `user_tools/` are scanned. Your tool is registered automatically — no code changes needed.

### Rules for action tools

- **Return JSON-serializable values**: dict, list, str, int, float, or bool. No custom objects.
- **No side effects outside the workspace**: tools that modify files should only touch the path given in `workspace`. Tools that call external APIs should be idempotent.
- **Handle errors gracefully**: if your tool might fail, catch exceptions and return `{"error": str(e)}` rather than raising. The subagent will continue and may retry differently.
- **No imports from coordinator, subagent, or memory**: tools are standalone. They cannot call back into the agent. Circular imports will cause startup failures.
- **Keep output bounded**: subagents include tool results in their context. A tool that returns 50KB of output will consume most of the context budget. Truncate to the first N lines if output could be large.

---

## Adding a Custom Test Oracle

The test oracle is the `kind="test"` tool that scores the agent's work. It is the single most important customization point — the agent's entire goal is to improve this score.

### Replace `user_tools/test.py`

```python
# user_tools/test.py
from tools.decorator import tool
import subprocess, sys, re

@tool(name="run_tests", description="Run the test suite and return a 0-1 score", kind="test")
def run_tests(workspace: str) -> dict:
    result = subprocess.run(
        [sys.executable, "-m", "pytest", workspace, "--tb=no", "-q"],
        capture_output=True, text=True, cwd=workspace,
    )
    output = result.stdout + result.stderr
    passed, total = _parse_output(output)
    if total == 0:
        return {"score": 0.0, "remark": "No tests found or all errored"}
    return {"score": passed / total, "remark": f"{passed}/{total} tests passed"}

def _parse_output(output):
    m = re.search(r"(\d+) passed", output)
    passed = int(m.group(1)) if m else 0
    m2 = re.search(r"(\d+) failed", output)
    failed = int(m2.group(1)) if m2 else 0
    return passed, passed + failed
```

### Contracts the oracle must satisfy

1. **Return dict with `score` key**: a float in `[0.0, 1.0]`. The coordinator compares `score > baseline` — if the score is always 0 or always 1, the agent cannot make progress.
2. **Return `remark` key** (optional): a human-readable explanation. Shown in the dashboard and stored in memory. The agent uses this to understand what improved.
3. **Accept `workspace` argument**: the path to the integration worktree. Run the tests there.
4. **Be deterministic**: if you call the oracle twice on the same code, it should return the same score. Non-deterministic oracles (flaky tests) will produce noisy learning signals.
5. **Use `kind="test"`**: this restricts the oracle to coordinator-only. Subagents cannot call it — they should not be able to observe the score during their work (opaque oracle design).

### Multi-metric scoring

You can compute a composite score from multiple signals:

```python
@tool(name="run_tests", ..., kind="test")
def run_tests(workspace: str) -> dict:
    pytest_score = _run_pytest(workspace)   # 0.0 – 1.0
    lint_score   = _run_linter(workspace)   # 0.0 – 1.0
    type_score   = _run_mypy(workspace)     # 0.0 – 1.0
    composite    = 0.7 * pytest_score + 0.2 * lint_score + 0.1 * type_score
    return {
        "score": composite,
        "remark": f"tests={pytest_score:.2f} lint={lint_score:.2f} types={type_score:.2f}",
    }
```

The coordinator only sees the composite score. The `remark` field carries the breakdown.

---

## Adding a Worker Model

### Step 1: Pull the model

```bash
ollama pull your-model:tag
```

### Step 2: Add it to `config.yaml`

```yaml
models:
  workers:
    - name:   "your-model:tag"
      skills: ["your-skill", "another-skill"]
      options:
        temperature: 0.4
        num_predict: 2048
```

### Step 3: Restart the server

`ModelRegistry.validate()` will check the model is available. If it is not pulled, startup fails with a clear error message including `ollama pull your-model:tag`.

### How to choose skills

Skills are free-form strings that the coordinator uses when decomposing hypotheses. The coordinator will tag subtasks with skills like `"code"`, `"refactor"`, `"math"`, `"sql"`, `"documentation"`. Your worker's skills should match these tags.

Check what skills the coordinator is assigning by watching the `model_routed` events in the dashboard. If a subtask requires `"sql"` but you have no SQL worker, you will see `fallback=true` in those events.

### Skill overlap gotcha

If two workers share skills (e.g. both have `"code"`), the router picks the first one in config order. There is no ranking by model quality. If you want a strong model to handle `"code"` tasks and a weaker one as a backup, list the stronger model first.

---

## Customizing Decomposition Behavior

The decomposer is controlled by the prompts in `coordinator/decomposer.py`.

### Changing the prompt

```python
# coordinator/decomposer.py
DECOMPOSE_SYSTEM = """..."""
```

The most common customizations:
- **Increasing subtask count**: change `"Use between 1 and {max_subagents} subtasks"` to a higher minimum.
- **Forcing scope**: add "Each subtask must name at least one file in `scope`."
- **Domain-specific instructions**: add "This is a Python codebase following PEP 8 and using pytest."

### Changing the JSON schema

The decomposer expects this schema from the model:

```json
{
  "subtasks": [
    {
      "goal": "...",
      "scope": ["file1.py", "file2.py"],
      "constraints": "...",
      "expected_output": "...",
      "required_skills": ["code", "refactor"]
    }
  ],
  "split_rationale": "..."
}
```

If you add fields to the schema, update `build_subtask_briefs()` to read them. If you rename fields, update `build_subtask_briefs()` and `shared/types.py:SubtaskBrief`. Breaking the JSON schema silently falls back to the single-subtask fallback — you will see single-subtask iterations in the dashboard if decomposition fails.

---

## Extending the Memory System

### Adding a field to iteration records

1. Add the field to `shared/types.py`:
   ```python
   @dataclass
   class IterationRecord:
       ...
       my_new_field: str | None = None
   ```
2. Update `memory/episodic.py`:
   ```python
   # init_db: add column
   """CREATE TABLE IF NOT EXISTS iterations (
       ...
       my_new_field TEXT,
       ...
   )"""
   # record: add to INSERT
   # _row_to_record: add to reconstruction
   ```
3. Write the field in `coordinator/coordinator.py` when creating `IterationRecord`.

The LanceDB schema does **not** need updating unless you want to embed the new field for retrieval.

### Changing what gets embedded

Currently, the hypothesis text is embedded. If you want to embed a richer representation (e.g. hypothesis + remark + score), change this in `memory/__init__.py`:

```python
async def record(self, record: IterationRecord) -> None:
    self._episodic.record(record)
    entry = MemoryEntry(
        text=f"{record.hypothesis} | score={record.score:.3f} | {record.remark}",
        ...
    )
    await self._semantic.store(entry)
```

Richer embeddings improve retrieval quality at the cost of slightly different similarity behavior.

### Adding a completely new memory type

If you want to store structured memory that is not just text (e.g. a graph of file dependencies), add a new layer:

1. Create `memory/graph.py` with a class that implements `store()` and `retrieve()`.
2. Instantiate it in `Memory.__init__()` and call it in `Memory.record()`.
3. Add a retrieval method to `Memory` and call it from `assemble_coordinator_context()`.

---

## Running the Test Suite

```bash
# From the project root
python -m pytest tests/ -v
```

If you get errors from ROS pytest plugins, use:

```bash
python3 -c "import pytest, sys; sys.exit(pytest.main(['tests/', '-v'], plugins=[]))"
```

This bypasses entrypoint plugin discovery and avoids ROS conflicts. See `setup.md` for full details.

### Writing tests

Tests live in `tests/`. The test structure mirrors the source:

```
tests/
├── test_types.py          — shared/types.py
├── test_memory.py         — memory/ (uses a temp directory)
├── test_router.py         — models/router.py
├── test_context.py        — subagent/context.py, coordinator/context.py
├── test_validator.py      — tools/validator.py
└── test_tools_runtime.py  — tools/runtime.py
```

For memory tests, use `pytest tmp_path` to get a clean directory:

```python
def test_episodic_record(tmp_path):
    mem = EpisodicMemory(str(tmp_path / "episodic.db"))
    mem.init_db()
    mem.record(IterationRecord(...))
    assert mem.count() == 1
```

---

## Development Tips

### Watching the event stream from the terminal

```bash
websocat ws://localhost:8000/events
```

`websocat` is a small CLI tool for WebSocket connections. Each line is one event.

### Running without a real Ollama server

Set `ollama_host` to a mock URL and stub the `OllamaClient` with a fixture. The test suite in `tests/` shows how to do this for unit tests.

### Debugging sandbox failures

Tool calls run in a subprocess. If a tool call returns `success=false` with a confusing error, add a `print(stderr)` line in `ToolRuntime.call()` to see the raw subprocess output:

```python
stdout, stderr = await proc.communicate(input=payload.encode())
if proc.returncode != 0:
    print("SANDBOX STDERR:", stderr.decode())
```

### Monitoring worktrees

```bash
git worktree list  # inside the target repo
```

If the agent is killed mid-iteration, orphaned worktrees may remain in `worktree_root`. They are safe to delete manually: `git worktree remove --force <path>`.

### Inspecting the episodic memory

```bash
sqlite3 data/episodic.db "SELECT iteration, outcome, score, hypothesis FROM iterations ORDER BY iteration DESC LIMIT 20;"
```

This gives you a quick view of recent iterations without the dashboard.
