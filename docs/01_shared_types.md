# Chapter 1 — Shared Data Types (`shared/types.py`)

## Why Read This First

Every module in the project passes data to other modules through typed dataclasses defined in `shared/types.py`. Before understanding how the coordinator forms hypotheses or how a subagent executes a subtask, you need to know what a `Hypothesis`, `SubtaskBrief`, and `SubtaskResult` actually contain. This file is the dictionary of the system.

---

## The File: `shared/types.py`

This module has no imports from the rest of the project. It only uses Python's standard library (`dataclasses`, `enum`, `uuid`, `datetime`). It is the one file with zero dependencies — everything else depends on it.

---

## Enumerations

### `TaskStatus`
Represents the final state of a subagent's run.

```python
class TaskStatus(str, Enum):
    SUCCESS = "success"   # subagent completed and returned a diff
    FAILED  = "failed"    # subagent explicitly reported failure
    PARTIAL = "partial"   # subagent hit the step cap without completing
    SKIPPED = "skipped"   # reserved for future use
```

Being a `str` enum means `TaskStatus.SUCCESS == "success"` is `True`, which makes JSON serialization and SQLite storage trivial.

### `OutcomeType`
Classifies the result of a full iteration.

```python
class OutcomeType(str, Enum):
    WIN      = "win"      # score improved vs. baseline
    MISTAKE  = "mistake"  # score did not improve (including crashes)
    NEUTRAL  = "neutral"  # reserved (e.g., unchanged diff)
```

The semantic memory stores both wins AND mistakes. Mistakes are especially important — they are the negative examples that prevent the coordinator from repeating failed approaches.

---

## Core Data Types

### `ModelSpec`
Describes one Ollama model and its configuration.

```python
@dataclass
class ModelSpec:
    name:    str             # Ollama model name, e.g. "qwen2.5-coder:7b"
    skills:  list[str]       # capability tags, e.g. ["code", "refactor"]
    options: dict[str, Any]  # inference params, e.g. {"temperature": 0.2}
```

`ModelSpec` objects are created by `ModelRegistry` at startup and passed around the system. The coordinator model, default model, and each worker model are all `ModelSpec` instances. When a subagent runs, `brief.model` is a `ModelSpec` and the Ollama client uses `spec.name` and `spec.options` automatically.

---

### `SubtaskBrief`
The instruction packet the coordinator gives to each subagent. Think of it as a work order.

```python
@dataclass
class SubtaskBrief:
    id:              str           # UUID, unique per iteration
    hypothesis_id:   str           # links back to the parent hypothesis
    goal:            str           # what the subagent must accomplish
    scope:           list[str]     # file/module paths it is allowed to touch
    constraints:     str           # hard rules (e.g. "do not change the API")
    expected_output: str           # what a successful result looks like
    required_skills: list[str]     # skill tags used for model routing
    model:           ModelSpec     # filled in by ModelRouter before dispatch
    matched_skills:  list[str]     # which skills triggered the routing decision
    fallback:        bool          # True if default model was used (no skill match)
```

**Important invariant:** Subtask briefs must be fully self-contained and independent. The coordinator writes them such that no subagent needs to know what another subagent is doing. This is enforced by the design: subagents run in separate Git worktrees with no shared live context.

The `model`, `matched_skills`, and `fallback` fields start empty and are filled in by `ModelRouter.select()` before the subagents are dispatched.

---

### `SubtaskResult`
What a subagent returns to the coordinator after finishing its work.

```python
@dataclass
class SubtaskResult:
    subtask_id:    str          # matches the brief's id
    diff:          str          # unified diff of all changes made
    files_touched: list[str]    # list of file paths modified
    summary:       str          # human-readable description of what was done
    status:        TaskStatus   # SUCCESS, FAILED, or PARTIAL
    error:         str | None   # exception message if status == FAILED
    steps_taken:   int          # how many ReAct steps were used
```

The `diff` field is the most important one. The coordinator reads it during the integration step and may accept, patch, or reject it. A crashed subagent produces a result with `status=FAILED` and `error` set — it never crashes the coordinator.

---

### `Hypothesis`
Represents one proposed improvement for the current iteration.

```python
@dataclass
class Hypothesis:
    id:        str            # UUID
    text:      str            # the hypothesis statement (1-3 sentences)
    rationale: str            # why the coordinator thinks this will work
    iteration: int            # which loop iteration this belongs to
    embedding: list[float]    # set by semantic memory for dup detection
```

A hypothesis is formed once per iteration and never changed within that iteration. If the anti-repetition gate rejects it, the coordinator reforms a *new* hypothesis and works with that instead.

---

### `IntegrationResult`
The output of the coordinator's review-and-merge step, representing the single unified change that will be tested.

```python
@dataclass
class IntegrationResult:
    hypothesis_id:     str
    diff:              str          # the final merged diff
    files_touched:     list[str]
    summary:           str          # what the merged change accomplishes
    path:              str          # path to the integration git worktree
    accepted_subtasks: list[str]    # subtask ids whose diffs were accepted
    rejected_subtasks: list[str]    # subtask ids whose diffs were rejected
```

The `path` field is critical: the `test` tool runs against this worktree, not against the main repo checkout. This isolates the test environment and prevents any changes from leaking into the main repo until after the score validates them.

---

### `MemoryEntry`
A single record in the semantic memory store.

```python
@dataclass
class MemoryEntry:
    id:        str
    text:      str              # the hypothesis text that was tried
    outcome:   OutcomeType      # WIN or MISTAKE
    score:     float            # the test score at the time
    remark:    str | None       # test oracle's remark (e.g. "60/80 passed")
    embedding: list[float]      # 768-dim vector (nomic-embed-text)
    ts:        datetime
    iteration: int
```

The semantic memory stores both `text` and `embedding`. When the coordinator retrieves "relevant past failures", it embeds the current hypothesis and queries for similar past `MemoryEntry` records. Entries with `outcome=MISTAKE` serve as **negative examples** — the coordinator is explicitly told "these were already tried and failed."

---

### `AgentState`
The minimal state needed to resume exactly where the process left off after a crash or restart.

```python
@dataclass
class AgentState:
    iteration:      int    # which iteration number we are on
    baseline_score: float  # current best score
    working_commit: str    # SHA of the commit to branch from
    run_id:         str    # UUID for this run session
```

This is written to SQLite (`state.db`) after every iteration. On restart, the coordinator loads this and continues the hypothesis chain from the same commit and score. No iteration is ever re-run.

---

### `IterationRecord`
The complete record of one iteration, written to the episodic memory log.

```python
@dataclass
class IterationRecord:
    id:                   str
    hypothesis:           str
    integrated_diff_hash: str             # SHA256[:16] of the merged diff
    subagent_contribs:    list[dict]      # per-subagent: {id, status, files}
    score:                float
    remark:               str | None
    outcome:              OutcomeType
    baseline_before:      float
    ts:                   datetime
    iteration:            int
```

This is the ground truth log. Every iteration — win or mistake — is appended. You can query this table to understand the full history of the run: which hypotheses were tried, what score they achieved, and whether the baseline improved.

---

## The `_new_id()` Helper

```python
def _new_id() -> str:
    return str(uuid.uuid4())
```

All dataclasses that need a unique ID use this as their default factory. IDs are UUIDs, never sequential integers, so records from different runs or different machines never collide.

---

## Summary for Contributors

| Type | Created by | Consumed by |
|------|-----------|------------|
| `ModelSpec` | `ModelRegistry` | `ModelRouter`, `OllamaClient`, `SubtaskBrief` |
| `SubtaskBrief` | `Coordinator.decompose()` | `Subagent.run()` |
| `SubtaskResult` | `Subagent.run()` | `Coordinator.review_and_integrate()` |
| `Hypothesis` | `Coordinator.form_hypothesis()` | `Coordinator.decompose()`, `Memory.record()` |
| `IntegrationResult` | `Coordinator.review_and_integrate()` | `Coordinator._run_test()`, `Memory.record()` |
| `MemoryEntry` | `Memory.record()` | `Coordinator.form_hypothesis()` (via `Memory.retrieve()`) |
| `AgentState` | `Coordinator._load_state()` | `Coordinator.__init__()`, resumed on restart |
| `IterationRecord` | `Coordinator._run_one_iteration()` | `Memory.record()`, `EpisodicMemory` |

If you want to add a new field that flows from the coordinator to a subagent, add it to `SubtaskBrief`. If you want to record something new about an iteration's outcome, add it to `IterationRecord`. Changes here ripple everywhere, so read through all the usages before modifying.
