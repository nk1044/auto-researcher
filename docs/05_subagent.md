# Chapter 5 — Subagent (`subagent/`)

## Overview

A subagent is a self-contained AI executor. It receives one `SubtaskBrief`, works in isolation inside a Git worktree, runs a loop of reasoning and tool-calling until the task is done, and returns one `SubtaskResult` to the coordinator.

The subagent knows nothing about other subagents, the coordinator's internal state, or the broader hypothesis. It just reads the brief and does the work.

---

## Files

```
subagent/
├── subagent.py    — the Subagent class and ReAct loop
└── context.py     — bounded context assembly for each ReAct step
```

---

## Git Worktree Isolation (`subagent.py`)

### Why Worktrees?

If multiple subagents all edited the same files in the same working directory, their changes would conflict immediately. Git worktrees solve this: each is a fully independent checkout of the same repository at the same commit, in a separate directory.

```
repo/
└── .git/
    └── worktrees/     ← git manages these
        ├── subagent-uuid1/   ← Subagent 1's private workspace
        ├── subagent-uuid2/   ← Subagent 2's private workspace
        └── integration-uuid/ ← Coordinator's integration workspace
```

Each subagent works in its own directory. Changes made by Subagent 1 are invisible to Subagent 2 until the coordinator explicitly merges their diffs.

### Creating a Worktree

```python
proc = await asyncio.create_subprocess_exec(
    "git", "worktree", "add", "--detach", worktree_path, baseline_commit,
    cwd=self.repo_path,
    ...
)
```

`--detach` means the worktree is not on any branch — it is directly at `baseline_commit`. This is important: it means the subagent's diffs are computed against the exact same baseline as every other subagent in this iteration, regardless of order.

### Getting the Diff

When the subagent finishes, the coordinator calls `get_diff()`:

```python
proc = await asyncio.create_subprocess_exec(
    "git", "diff", "HEAD",
    cwd=self._worktree_path,
    ...
)
```

`git diff HEAD` in a detached worktree shows everything that changed relative to `baseline_commit`. This is the diff that the coordinator integrates.

---

## The ReAct Loop (`subagent.py`)

ReAct (Reason + Act) is the pattern where an AI model alternates between reasoning steps ("I need to read the file first") and action steps (calling a tool to read the file). This loop continues until the task is complete.

### System Prompt

```python
SYSTEM_PROMPT = """You are a focused software engineering subagent. Your job is to
complete exactly one subtask by reading and editing files in your assigned workspace.
You work independently — you cannot communicate with other agents.

Rules:
- Only touch files within your assigned scope.
- Do NOT call the 'test' or 'save' tools — those are coordinator-only.
- When you have completed the task, respond with DONE followed by a summary.
- If you cannot complete the task, respond with FAILED followed by the reason.
- Be precise and make minimal changes that accomplish the goal.
"""
```

The system prompt is short and tight. It does not describe the broader system or other agents — the subagent does not need that context.

### The Loop

```python
while step < self.step_cap:          # step_cap default: 20
    # 1. Regenerate rolling summary every N steps
    if step > 0 and step % self.summary_every_n == 0:
        self._rolling_summary = await self._regenerate_summary()

    # 2. Assemble bounded context from scratch
    context = assemble_subagent_context(
        brief, tool_schemas, memory_entries, self._rolling_summary,
        token_budget, ratios
    )

    # 3. Build messages: system + context + abbreviated step history
    messages = [system_message, context_message, *last_5_history]

    # 4. Call the model
    response = await self.client.chat(
        model_spec=self.brief.model,
        messages=messages,
        tools=tool_schemas,
    )

    # 5. Check for terminal conditions
    if response.content.upper().startswith("DONE"):
        return SubtaskResult(status=SUCCESS, diff=get_diff(), ...)
    if response.content.upper().startswith("FAILED"):
        return SubtaskResult(status=FAILED, error=reason)

    # 6. Execute any tool calls
    if response.tool_calls:
        result = await self._execute_tool_calls(response.tool_calls)
        self._steps_history.append({"action": content, "result": result})
    else:
        self._steps_history.append({"action": content, "result": "(no tool call)"})

    step += 1

# Reached step_cap without DONE/FAILED
return SubtaskResult(status=PARTIAL, diff=get_diff(), ...)
```

### Terminal Conditions

The loop ends in one of three ways:

1. **`DONE`**: The model says it is done. The subagent reads the final diff and returns `SUCCESS`.
2. **`FAILED`**: The model says it cannot complete the task. Returns `FAILED` with the reason.
3. **Step cap reached**: The loop ran `step_cap` times without finishing. Returns `PARTIAL` with whatever diff was produced so far.

The coordinator handles `PARTIAL` results — it may accept partial changes during integration if they improve things.

### Step History Truncation

The full step history (`self._steps_history`) is maintained but only the **last 5 steps** are included in each context message. This prevents the context from growing unboundedly.

Every `summary_every_n` steps, the model is asked to produce a new rolling summary of its progress:

```python
async def _regenerate_summary(self) -> str:
    prompt = regenerate_rolling_summary_prompt(self.brief, self._steps_history)
    response = await self.client.chat(model_spec=self.brief.model, messages=[...])
    return response.content.strip()
```

The rolling summary replaces the full step history in the context — it compresses what happened into a few sentences, preventing context overflow on long tasks.

---

## Tool Execution

When the model returns tool calls, the subagent executes them sequentially:

```python
async def _execute_tool_calls(self, tool_calls: list[dict]) -> dict:
    for tc in tool_calls:
        name = tc["function"]["name"]
        args = tc["function"]["arguments"]
        # Inject workspace path if not already present
        if "workspace" not in args:
            args["workspace"] = self._worktree_path
        # Call with caller="subagent" (enforces kind restrictions)
        result = await self.tools.call(name, caller="subagent", **args)
```

Notice `caller="subagent"` — this is how the tool runtime enforces that subagents cannot call the `test` or `save` tools.

The `workspace` injection is a convenience: most file-operation tools need to know which directory to work in. Rather than the model having to include the path in every tool call, the subagent injects the worktree path automatically.

---

## Context Assembly (`subagent/context.py`)

### Budget Allocation

```
Total context budget: 8192 tokens (configurable)

Section         Ratio   Chars budget  Purpose
─────────────   ─────   ────────────  ──────────────────────────────
task_brief      20%     ~4,915        The SubtaskBrief + tool schemas
files           50%     ~12,288       In-scope file contents (fresh from disk)
memory          15%     ~3,686        Relevant past iterations
rolling_summary 15%     ~3,686        Rolling summary of current progress
```

The character budget uses `CHARS_PER_TOKEN = 3` (a conservative approximation for code-heavy content).

### `assemble_subagent_context()`

```python
def assemble_subagent_context(
    brief: SubtaskBrief,
    tool_schemas: list[dict],
    memory_entries: list[MemoryEntry],
    rolling_summary: str,
    token_budget: int,
    ratios: dict[str, float],
) -> str:
```

This function is called at the start of every ReAct step. It rebuilds the context completely from scratch — no accumulation, no stale content. The sections are:

**Section 1 — Task Brief + Tool Schemas**

The subtask brief is formatted as a structured block:
```
## Subtask Brief
ID: <uuid>
Goal: <goal text>
Scope (files/modules): src/foo.py, src/bar.py
Constraints: <constraints>
Expected output: <expected output>
Required skills: code, refactor

## Available Tools
[JSON tool schemas]
```

**Section 2 — In-Scope File Slices (fresh from disk)**

```python
for path in brief.scope:
    content = read_file_slice(path, per_file_chars)
    sections.append(f"### {path}\n```\n{content}\n```")
```

Each file is read fresh from the worktree on every step. This means the context always reflects the current state of the files — if the subagent edited a file in the previous step, the next step sees the edited version. This is what grounds the model in reality and prevents it from working from a stale mental model.

**Section 3 — Retrieved Memory**

A small number (3) of relevant past iterations retrieved from semantic memory. These are shown with their outcome label:
```
- [WIN score=0.823 — 66/80 tests passed]: refactored the sort algorithm...
- [ALREADY TRIED, FAILED — broke imports]: tried to inline the helper function...
```

**Section 4 — Rolling Summary**

The most recent rolling summary. Empty on the first step ("starting fresh"), updated every `summary_every_n` steps.

### Truncation, Not Dropping

If a section exceeds its budget, it is **truncated** (characters are cut off with an explicit `... [truncated]` marker), never silently dropped. This means the model always sees at least the beginning of every section — the most important information is at the top.

---

## The Subagent's Lifecycle

```
Coordinator.dispatch()
    │
    ├── Subagent.__init__(brief, baseline_commit, ...)
    │
    ├── Subagent.run()
    │   ├── _create_worktree()    ← git worktree add
    │   └── _react_loop()
    │       ├── step 0: assemble context → chat() → [DONE | tool call | reasoning]
    │       ├── step 1: assemble context → chat() → [DONE | FAILED | tool call]
    │       ├── step N: summary regenerated every summary_every_n steps
    │       └── → SubtaskResult(diff, files, summary, status)
    │
    └── (worktree cleanup happens in coordinator._cleanup_worktrees())
```

The worktree is NOT cleaned up inside `Subagent.run()` — it is cleaned up by the coordinator after it has read the diff. If the subagent cleaned up its own worktree, the coordinator could not read the diff.

---

## For Contributors

**Increasing subagent capability:**
- Increase `subagent_step_cap` in `config.yaml` for more complex tasks.
- Decrease `summary_every_n` to regenerate summaries more often (more tokens spent, better continuity).
- Add more action tools to `user_tools/` so subagents have more ways to interact with the code.

**Debugging a subagent:**
- All `subagent_progress` and `subagent_done` events appear in the dashboard event log.
- The `steps_taken` field in `SubtaskResult` tells you how many steps were used.
- A `PARTIAL` result (step cap hit) usually means the task was too complex — either simplify the hypothesis decomposition or increase the step cap.

**Changing context ratios:**
Edit `subagent_context_ratios` in `config.yaml`. The ratios must sum to 1.0. Increasing `files` gives the model more code to work with; increasing `memory` gives it more historical context.
