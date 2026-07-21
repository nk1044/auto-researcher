# Chapter 4 — Tools System (`tools/`)

## Overview

Tools are the hands of the agent. They let the coordinator and subagents interact with the real world: reading and writing files, running shell commands, executing tests, and saving results to GitHub.

The tools system has four concerns:

1. **Declaration** (`decorator.py`) — how you register a function as a tool and auto-generate its JSON schema.
2. **Execution** (`runtime.py`) — how tools are called safely in a sandboxed subprocess.
3. **Validation** (`validator.py`) — how diffs are checked for reward-hacking.
4. **Persistence** (`save_tool.py`) — how improvements are committed and pushed to GitHub.

---

## Tool Declaration (`tools/decorator.py`)

### The `@tool` Decorator

```python
from tools.decorator import tool

@tool(name="run_tests", description="Run the test suite on workspace", kind="test")
def run_tests(workspace: str) -> dict:
    ...
    return {"score": 0.75, "remark": "60/80 tests passed"}
```

The decorator does three things:
1. Inspects the function's type hints.
2. Generates an Ollama-compatible JSON schema (the format Ollama uses for tool calling).
3. Attaches a `ToolSchema` object to the function as `.schema`.

### Schema Generation

Python type hints are mapped to JSON schema types:

| Python | JSON Schema |
|--------|------------|
| `str`  | `"string"` |
| `int`  | `"integer"` |
| `float` | `"number"` |
| `bool` | `"boolean"` |
| `list` | `"array"` |
| `dict` | `"object"` |
| `Optional[X]` | type of X, not required |

A parameter is **required** if it has no default value and is not `Optional`. The generated schema looks like:

```json
{
  "type": "function",
  "function": {
    "name": "run_tests",
    "description": "Run the test suite on workspace",
    "parameters": {
      "type": "object",
      "properties": {
        "workspace": {"type": "string"}
      },
      "required": ["workspace"]
    }
  }
}
```

This is exactly the format Ollama's tool-calling API expects.

### `ToolSchema` Dataclass

```python
@dataclass
class ToolSchema:
    name:        str        # tool name used in tool calls
    description: str
    kind:        str        # "action" | "test" | "save"
    parameters:  dict       # the full Ollama-format schema dict
    fn:          Callable   # the actual function to call
    source_file: str        # absolute path to the .py file
```

The `source_file` field is critical for the sandbox (explained below). It holds the absolute path to the `.py` file where the function is defined. When the sandbox subprocess needs to run the tool, it imports the function directly from this file — it does not rely on `sys.modules` or Python's import system.

### The `auto_schema()` Function

If you define a function in `user_tools/` without the `@tool` decorator, `auto_discover()` calls `auto_schema()` to generate a schema automatically using the function name and docstring. This is a convenience for quick tools. The kind defaults to `"action"`.

For production tools, always use `@tool` explicitly — it documents the kind and description clearly.

---

## Tool Runtime (`tools/runtime.py`)

### Tool Registration

```python
tools = ToolRuntime(config)
tools.register(save_to_github)    # register a specific function
tools.auto_discover("user_tools") # import all .py files in a directory
```

`auto_discover()` does the following for each `.py` file:
1. Loads it as a dynamic module with `importlib.util.spec_from_file_location`.
2. Iterates over the module's attributes.
3. For each callable defined in that file (not imported from elsewhere), registers it.
4. If the function has `@tool`, uses its pre-built `ToolSchema`. If not, calls `auto_schema()`.

The filter `getattr(obj, "__module__", None) != module_name` ensures imported functions (like `os.path.join`) are not accidentally registered as tools.

### Tool Kinds and Access Control

```python
_RESTRICTED_KINDS = frozenset({"test", "save"})
```

When `ToolRuntime.call()` is invoked, it checks the `caller` argument:
- `caller="coordinator"`: can call any tool.
- `caller="subagent"`: calling a `"test"` or `"save"` tool returns `ToolResult(success=False, error="Restricted: test/save are coordinator-only")`.

This is a soft enforcement layer. The stronger enforcement is that subagents are never given the schemas for test/save tools in the first place:

```python
# Subagent only gets "action" tool schemas
tool_schemas = self.tools.get_schemas(kinds=["action"])
```

So an LLM-driven subagent literally cannot call the test tool — it does not know it exists.

### Sandboxed Execution

Every tool call runs in a **fresh subprocess**:

```python
proc = await asyncio.create_subprocess_exec(
    sys.executable, "-c", sandbox_script,
    stdin=asyncio.subprocess.PIPE,
    stdout=asyncio.subprocess.PIPE,
    stderr=asyncio.subprocess.PIPE,
)
stdout, stderr = await proc.communicate(input=payload.encode())
```

The payload (kwargs as JSON) is written to the subprocess's stdin. The result (success/failure as JSON) is read from stdout.

**Why a subprocess?**
- Isolation: a tool that crashes (segfault, exit()) does not crash the coordinator.
- Resource limits: the subprocess sets `resource.RLIMIT_CPU` and `resource.RLIMIT_AS` before calling the function.
- Timeout: `asyncio.wait_for()` kills the subprocess if it runs too long.

### The Sandbox Script

`_build_sandbox_script()` generates a self-contained Python string that:

1. Adds the project root to `sys.path` (so tool files can `import tools.decorator`).
2. Reads the payload JSON from stdin.
3. Sets resource limits:
   ```python
   resource.setrlimit(resource.RLIMIT_CPU, (cpu_limit, cpu_limit))
   resource.setrlimit(resource.RLIMIT_AS, (mem_bytes, mem_bytes))
   ```
4. Loads the tool function from its `source_file` using `importlib.util.spec_from_file_location`.
5. Calls the function (using `asyncio.run()` if it's async).
6. Writes `{"success": true, "value": result}` or `{"success": false, "error": "..."}` to stdout.

**Why load from `source_file` rather than by module name?**

When `auto_discover()` loads a file dynamically, it gives the module a synthetic name like `_auto_discover_test`. This name only exists in the parent process's `sys.modules`. The subprocess has no knowledge of this name. Loading from the absolute file path (`source_file`) bypasses this problem completely.

### `ToolResult`

```python
@dataclass
class ToolResult:
    success: bool
    value:   Any         # the function's return value on success
    error:   str | None  # exception message on failure
```

The coordinator always checks `result.success` before using `result.value`. A failed tool call never crashes the coordinator — it is logged and typically recorded as a mistake in memory.

---

## Diff Validator (`tools/validator.py`)

### Purpose

Before the coordinator calls `save`, it validates the integrated diff to prevent reward hacking. Reward hacking is when an agent cheats by modifying the test harness to make itself look better.

### `validate_diff(diff, repo_path, protected_patterns)`

```python
def validate_diff(
    diff: str,
    repo_path: str,
    protected_patterns: list[str],
) -> tuple[bool, str]:
    # Returns (is_valid: bool, reason: str)
```

Four checks are performed in order:

**1. Non-empty:** An empty diff is always invalid. It means the integration produced no changes, which means there is nothing to test or save.

**2. Touches at least one file:** The diff must have recognizable `+++ b/...` headers. A diff that somehow has no file modifications is structurally invalid.

**3. No protected paths:** Every file touched by the diff is checked against `protected_patterns`. The default patterns are:
```
tests/, test/, held_out/, eval/, benchmark/, *.test.*, test_*.py, *_test.py
```
If any touched file matches a pattern, the diff is rejected with a message naming the offending path.

**4. Valid hunk headers:** The diff must contain at least one `@@ -N,N +N,N @@` header. Without hunk headers, `git apply` would fail — this check catches truncated or malformed diffs early.

### `extract_touched_files(diff)`

Parses `+++ b/...` lines from the diff. Returns a list of file paths with the `b/` prefix stripped. Skips `/dev/null` (which appears for newly created files on the old-file side).

### Why This Matters

An agent optimizing an opaque score has an incentive to find shortcuts. The most dangerous shortcut is modifying the test harness to report a higher score regardless of code quality. The validator blocks this by refusing any diff that touches test files. If you write a new evaluation in `tests/`, add it to `protected_patterns` in `config.yaml`.

---

## Save Tool (`tools/save_tool.py`)

### Purpose

When the score improves and the diff passes validation, the coordinator commits the change and pushes it to GitHub. This is the save tool.

### `save_to_github(diff, repo_path, remote, branch_prefix, iteration, meta)`

```python
@tool(name="save_to_github", description="Commit integrated diff and push to GitHub", kind="save")
async def save_to_github(
    diff: str,
    repo_path: str,
    remote: str,
    branch_prefix: str,
    iteration: int,
    meta: dict,
) -> dict:
    # Returns {"branch": "...", "commit": "...", "url": "..."}
```

**Step-by-step:**

1. Build a branch name: `f"{branch_prefix}/{iteration:04d}"` → e.g. `auto-researcher/0042`
2. `git checkout -B <branch>` — create or reset the branch at current HEAD.
3. `git apply --index -` — apply the diff from stdin. The `--index` flag stages the changes immediately.
4. `git commit -m <message>` — commit with a message that includes the hypothesis and score.
5. `git rev-parse HEAD` — capture the new commit SHA.
6. `git push <remote> <branch>` — push to GitHub.
7. `git remote get-url <remote>` — fetch the remote URL for the return value.

Every git command is run with `asyncio.create_subprocess_exec` — never `subprocess.run`, which would block the event loop.

### Error Handling

Any non-zero git return code raises `RuntimeError` with the git stderr output. The coordinator catches this in `_maybe_save()`, logs it as a warning, and continues the loop. A save failure is not a fatal error — the baseline is not updated and the next iteration tries a new hypothesis.

### Branch Naming Convention

Each saved improvement gets its own branch: `auto-researcher/0001`, `auto-researcher/0002`, etc. This creates a clear history in GitHub: you can browse these branches to see the evolution of the codebase iteration by iteration. The `github_branch_prefix` in config can be changed to namespace runs from different experiments.

---

## User Tools (`user_tools/`)

### `test.py` — The Test Oracle

```python
@tool(name="run_tests", description="Run the test suite and return a score", kind="test")
def run_tests(workspace: str) -> dict:
    # Returns {"score": float, "remark": str | None}
```

The sample implementation runs `pytest --tb=no -q` in the workspace directory and computes `score = passed / total`. Replace this with your own evaluation logic. The only contract is the return format: `{"score": float, "remark": str | None}`.

The `workspace` argument is always the path to the integration worktree. The test runs there, not in the main repo — so the tested code includes the proposed changes but the test files are unchanged.

### `sample_action.py` — A Sample Action Tool

```python
@tool(name="run_shell", description="Run a shell command in workspace", kind="action")
def run_shell(workspace: str, command: str) -> dict:
    # Returns {"stdout": str, "stderr": str, "returncode": int}
```

This is an example of a custom action tool. It runs a shell command in the workspace and returns the output. Subagents can call action tools during their ReAct loop. Add your own tools here — static analysis, linters, dependency checks, whatever your workflow needs.

---

## For Contributors: Adding a Custom Tool

1. Create a new `.py` file in `user_tools/`:

```python
# user_tools/my_linter.py
from tools.decorator import tool

@tool(name="run_linter", description="Run ruff on workspace and return error count", kind="action")
def run_linter(workspace: str) -> dict:
    import subprocess
    result = subprocess.run(["ruff", "check", workspace], capture_output=True, text=True)
    errors = result.stdout.count("\n")
    return {"errors": errors, "output": result.stdout[:500]}
```

2. Restart the server. `auto_discover("user_tools")` will pick it up automatically.

3. Subagents will see `run_linter` in their tool schemas and can call it during their ReAct loop.

**Rules for writing tools:**
- Return a JSON-serializable value (dict, list, str, int, float, bool).
- Use `kind="test"` only for the test oracle (coordinator-only).
- Use `kind="save"` only for save operations (coordinator-only).
- Use `kind="action"` for everything else (callable by subagents too).
- Keep tools stateless — each call should be idempotent.
- Do not import from `coordinator`, `subagent`, or `memory` — tools are standalone.
