# Chapter 6 — Coordinator (`coordinator/`)

## Overview

The coordinator is the brain of the system. It:
- Runs the infinite loop.
- Forms hypotheses.
- Decomposes them into subtasks and routes each to the right model.
- Dispatches subagents in parallel.
- Reviews and integrates their results.
- Runs the test oracle exactly once.
- Writes everything to memory.
- Saves to GitHub if the score improved.
- Repeats forever.

The coordinator is a single actor — there is only ever one. It is the only component allowed to call the `test` and `save` tools.

---

## Files

```
coordinator/
├── coordinator.py   — the Coordinator class and infinite loop (main entry)
├── context.py       — context assembly for hypothesis formation and integration
├── decomposer.py    — prompts and helpers for decomposing a hypothesis
└── integrator.py    — prompts and helpers for merging subagent diffs
```

---

## The Infinite Loop (`coordinator/coordinator.py`)

### The Loop

```python
async def run(self) -> None:
    await self._load_state()
    await aemit(EventType.LOOP_STARTED, {...})

    while not self.stop_requested:
        await self.pause_gate.wait()    # blocks on pause; never exits
        if self.stop_requested:
            break
        try:
            await self._run_one_iteration()
        except Exception as exc:
            logger.exception("Unhandled error in iteration %d: %s", ...)
            await aemit(EventType.ERROR, {...})
            # loop continues unconditionally

    await self._drain_and_flush()
```

Three things to notice:

1. `while not self.stop_requested` — the only exit condition.
2. `await self.pause_gate.wait()` — when `pause()` is called, `pause_gate` is cleared and this blocks indefinitely. When `resume()` is called, `pause_gate` is set and the loop unblocks. The current iteration runs to completion before a pause takes effect.
3. `return_exceptions=True` logic — the outer try/except catches any unhandled error from an iteration. The loop continues. A bad iteration is logged and recorded as a mistake, but it never terminates the process.

### One Iteration

```python
async def _run_one_iteration(self) -> None:
    n = self.state.iteration

    # 1. Form hypothesis
    hyp = await self.form_hypothesis()

    # 2. Anti-repetition gate
    is_dup = await self.memory.is_duplicate_failure(hyp.text)
    if is_dup:
        hyp = await self.reform_with_novelty(hyp)

    # 3. Decompose + route
    briefs = await self.decompose(hyp)
    for b in briefs:
        model_spec, matched, fallback = self.router.select(b.required_skills)
        b.model = model_spec
        b.matched_skills = matched
        b.fallback = fallback
        await aemit(EventType.MODEL_ROUTED, {...})

    # 4. Dispatch subagents in parallel
    results = await self._dispatch_subagents(briefs)
    ok_results = [r for r in results if r is not None and not isinstance(r, BaseException)]

    # 5. Review + integrate
    integrated = await self.review_and_integrate(hyp, ok_results)

    # 6. Test once
    score, remark = await self._run_test(integrated.path)

    # 7. Record
    record = IterationRecord(hypothesis=hyp.text, score=score, ...)
    await self.memory.record(record)

    # 8. Save on improvement
    if score > self.baseline and integrated.diff.strip():
        saved = await self._maybe_save(integrated, hyp, score, n)
        if saved:
            self.baseline = score
            await self._advance_working_commit(integrated.path)

    # 9. Cleanup worktrees
    await self._cleanup_worktrees(results, integrated.path)

    # 10. Advance iteration counter and persist state
    self.state.iteration += 1
    self.state.baseline_score = self.baseline
    self.memory.save_state(self.state)
```

Notice step 10: `save_state()` is called after every iteration, not just on success. This ensures resume always works correctly.

---

## Hypothesis Formation (`coordinator/coordinator.py`)

### Context Assembly

Before calling the LLM, the coordinator assembles a rich context from four sources:

```python
context = assemble_coordinator_context(
    task_spec=self._task_spec(),           # 20% — what we're trying to improve
    tool_schemas=[],                        # (no tools during hypothesis formation)
    memory_wins=wins,                       # 30% — past successes for inspiration
    memory_failures=failures,              # 30% — past failures to avoid
    file_slices=file_slices,               # 35% — current code state
    rolling_summary=self._iteration_rolling_summary,  # 15%
    token_budget=self.token_budget,
    ratios=self.coord_ratios,
)
```

The `_load_file_slices()` method reads the 8 most recently modified Python files from the target repo. "Recently modified" is a useful heuristic — recent changes are likely to be the most relevant to the current improvement hypothesis.

### The Prompt

```python
HYPOTHESIS_SYSTEM = """You are a scientific software engineering coordinator.
Your job is to form ONE concrete, testable hypothesis about how to improve the
target codebase's test pass-rate score.

Rules:
1. The hypothesis must be specific and actionable (name files/functions to change).
2. It must NOT repeat a previously failed hypothesis.
3. Ground it in the retrieved memory and current code state.
4. Output ONLY JSON: {"hypothesis": "...", "rationale": "...", "target_files": [...]}
"""
```

The model is asked for **one hypothesis**, not a list. This is deliberate. Multiple hypotheses would require a selection step; a single forced choice focuses the model and keeps the iteration tight.

### JSON Extraction

The response is expected to be JSON, but LLMs sometimes wrap it in markdown code blocks or add preamble text. The `_extract_json()` helper in `decomposer.py` handles this:

```python
def _extract_json(text: str) -> dict:
    try:
        return json.loads(text)           # try direct parse
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{[\s\S]*\}", text)  # find first JSON object
    if match:
        return json.loads(match.group())
    raise ValueError(...)
```

If even this fails, the coordinator falls back to treating the entire response as the hypothesis text. The system should keep working even if the model produces imperfect output.

---

## Anti-Repetition Gate

### The Problem

Without a gate, the coordinator might re-form the same hypothesis repeatedly — especially when improvements are hard to find and the model gravitates toward the same approach. Trying the same failed approach wastes an entire iteration (including subagent compute, test execution, and memory writes).

### The Check

```python
is_dup = await self.memory.is_duplicate_failure(hyp.text)
```

This embeds the new hypothesis and searches for similar `MISTAKE` entries in semantic memory. If the nearest failure has cosine similarity > `dup_threshold` (0.92), the hypothesis is rejected.

### Reform with Novelty

```python
NOVELTY_SYSTEM = """...The previous hypothesis was too similar to a past failure.
Form a NOVEL hypothesis that avoids all previously tried approaches..."""

NOVELTY_USER = """Rejected hypothesis: {rejected_hypothesis}
Past failures to AVOID:
{past_failures}

Form a NOVEL hypothesis..."""
```

The coordinator is given the rejected hypothesis and the top-k past failures as explicit negative examples. It is also given a higher temperature (base temperature + `novelty_boost = 0.3`) to encourage more creative responses.

This is called **bounded retries** — there is an implicit retry built into the iteration (one reform attempt), not an unbounded loop. If the reformed hypothesis also turns out to be a duplicate, it still proceeds (the gate only runs once per iteration).

---

## Decomposition (`coordinator/decomposer.py`)

### The Prompt

```python
DECOMPOSE_SYSTEM = """...decompose it into independent subtasks that can be
executed in parallel.

Rules:
1. Each subtask must be fully self-contained — subagents cannot communicate.
2. Subtasks must not have cross-dependencies.
3. Use between 1 and {max_subagents} subtasks.
4. Each subtask must specify: goal, scope, constraints, expected_output, required_skills.

Output ONLY valid JSON matching this schema: {"subtasks": [...], "split_rationale": "..."}
"""
```

The coordinator is instructed to produce a specific JSON schema. The key constraints the model must follow:

- **Independence**: no subtask can depend on another's output. Subagents do not communicate.
- **Scope assignment**: each subtask must name the files it will touch. This prevents two subagents from touching the same file (which would create a merge conflict).
- **Skill tagging**: each subtask must declare `required_skills` so the router can assign the right model.

### Adaptive Decomposition

The number of subtasks (1 to `max_subagents`) is chosen by the coordinator per hypothesis. A trivial hypothesis ("add a docstring to this function") gets 1 subtask. A complex one ("refactor the authentication module to use the new token format") might get 3 or 4.

The coordinator also chooses between two decomposition styles:
- **Scope-based**: split by files ("Subtask 1: edit auth.py, Subtask 2: edit tests.py" — note: test files are protected, so this would not save).
- **Task-based**: split by concern ("Subtask 1: refactor the token parser, Subtask 2: update the validators").

### Fallback

If JSON parsing fails, the decomposer falls back to a single subtask covering the entire hypothesis:

```python
decomposition = {
    "subtasks": [{
        "goal": hyp.text,
        "scope": [],
        "constraints": "Do not modify test files.",
        "expected_output": "Improved code with no regressions",
        "required_skills": ["code"],
    }]
}
```

This ensures the loop never stalls on a parsing failure.

---

## Dispatching Subagents

```python
async def _dispatch_subagents(self, briefs: list[SubtaskBrief]) -> list[...]:
    async def run_one(brief: SubtaskBrief) -> SubtaskResult:
        async with self._sem:   # semaphore limits concurrency
            agent = Subagent(
                brief=brief,
                baseline_commit=self.state.working_commit,
                ...
            )
            return await agent.run()

    results = await asyncio.gather(
        *[run_one(b) for b in briefs],
        return_exceptions=True,  # ← a crashed subagent never kills the loop
    )
    return list(results)
```

`asyncio.gather(return_exceptions=True)` is the key safety net: if a subagent raises an exception, it is returned as a `BaseException` object in the results list instead of propagating. The coordinator checks `isinstance(r, BaseException)` and treats that subagent as failed.

The semaphore limits how many subagents run concurrently. With `max_subagents=4`, at most 4 subagents are active at once. If you have 6 subtasks, the first 4 start immediately; the remaining 2 wait for a slot.

---

## Integration (`coordinator/integrator.py`)

### Purpose

The coordinator receives diffs from multiple subagents. These diffs may:
- Overlap (two subagents modified the same file, possibly the same lines).
- Conflict (one subagent added a function that another moved).
- Be empty (a subagent did nothing).
- Be partial (a subagent only completed half the task).

The integration step merges these into a single coherent diff that `git apply` can apply cleanly.

### LLM-Guided Integration

The preferred path asks the coordinator model to review and merge:

```python
REVIEW_SYSTEM = """You are a senior software engineer reviewing parallel subagent
contributions...

Output JSON:
{
  "decisions": [{"subtask_id": "...", "decision": "ACCEPT|PATCH|REJECT", "reason": "..."}],
  "merged_diff": "...unified diff string...",
  "summary": "..."
}
"""
```

The coordinator model is shown all subagent diffs and asked to:
1. Accept, patch, or reject each subtask's contribution.
2. Produce a single unified diff that merges all accepted contributions.
3. Resolve any conflicts.

This is a genuine LLM reasoning step — the coordinator reads multiple diffs and produces one. It is not a blind `git merge`.

### Naive Merge Fallback

If the LLM produces an unparseable response, or if the merged diff fails to apply with `git apply`, the system falls back to **naive sequential merge**:

```python
async def naive_merge(results, integration_worktree) -> tuple[str, list[str], list[str]]:
    for r in results:
        ok = await apply_diff_to_worktree(r.diff, integration_worktree)
        if ok:
            accepted.append(r.subtask_id)
        else:
            rejected.append(r.subtask_id)  # conflict — skip
```

Each diff is applied in order. If applying a diff conflicts with an already-applied change, it is rejected. This is a best-effort merge that maximises the amount of accepted work, even if some subtasks' contributions are lost.

### Integration Worktree

The integration result is a real Git worktree — not a diff string. The `test` tool runs against this worktree path. This ensures:
- The test runs on actually-applied code, not a theoretical diff.
- If the diff fails to apply cleanly, the test would catch it anyway.
- The coordinator can call `git diff HEAD` on the integration worktree to get the final authoritative diff before saving.

---

## Testing and Saving

### `_run_test()`

```python
async def _run_test(self, workspace: str) -> tuple[float, str | None]:
    result = await self.tools.call("run_tests", caller="coordinator", workspace=workspace)
    if result.success and isinstance(result.value, dict):
        score = float(result.value.get("score", 0.0))
        remark = result.value.get("remark")
        return score, remark
    return 0.0, f"test error: {result.error}"
```

The test is called **once per iteration**, after integration, on the integration worktree. A score of 0.0 is returned on any failure — the coordinator never crashes on a test failure.

### `_maybe_save()`

```python
if score > self.baseline and integrated.diff.strip():
    # 1. Validate the diff (reward-hack guard)
    valid, reason = validate_diff(integrated.diff, ...)
    if not valid:
        emit(REWARD_HACK_REJECTED, {reason: reason})
        return False

    # 2. Call the save tool
    result = await self.tools.call("save_to_github", caller="coordinator", ...)
    if result.success:
        emit(SAVED, {branch: ..., score: ...})
        return True
```

The save is gated on three conditions:
1. `score > self.baseline` — strictly better, never equal.
2. `integrated.diff.strip()` — actually made changes.
3. `validate_diff()` — does not touch protected files.

---

## Context Assembly (`coordinator/context.py`)

### Two Assembly Functions

**`assemble_coordinator_context()`** — used during hypothesis formation.

Budget: 20% task spec, 30% memory, 35% file slices, 15% rolling summary.

The memory section is split into two subsections: wins (labeled `[WIN score=0.82]`) and failures (labeled `[ALREADY TRIED, FAILED]`). The explicit labeling ensures the model does not confuse past failures with past successes.

**`assemble_integration_context()`** — used during the review-and-integrate step.

Budget: 25% hypothesis, 40% subagent results (including diffs), 35% baseline file slices.

The subagent results section shows each diff with its status, summary, and files — enough information for the coordinator to make an intelligent merge decision.

---

## Coordinator State

```python
class Coordinator:
    stop_requested: bool          # set by stop() — exits the loop
    pause_gate: asyncio.Event     # cleared by pause(), set by resume()
    state: AgentState             # iteration, baseline, commit
    baseline: float               # current best score
    _sem: asyncio.Semaphore       # limits concurrent subagents
    _current_hypothesis: Hypothesis
    _iteration_rolling_summary: str
```

`stop_requested` and `pause_gate` are the two control surfaces exposed to the REST API. `stop()` sets `stop_requested` and also calls `pause_gate.set()` to unblock a paused loop so it can exit cleanly.

---

## For Contributors

**Changing decomposition logic:**
Edit the prompts in `coordinator/decomposer.py`. The JSON schema must be preserved — changing the key names requires updating `build_subtask_briefs()`.

**Adding a new hypothesis source:**
`form_hypothesis()` currently only queries semantic memory. You could add a call to `get_recent_iterations()` to include the last N iterations' hypotheses and scores directly, giving the model a clearer picture of recent progress.

**Changing integration logic:**
The integration step is the most fragile part of the system. If you change the prompts in `coordinator/integrator.py`, make sure to test with overlapping diffs — the naive merge fallback exists precisely because LLM integration can fail.

**Monitoring hypothesis quality:**
All `hypothesis_formed` and `dup_rejected` events appear in the dashboard. If you see many `dup_rejected` events, the `dup_threshold` may be too low — lower it to 0.85 to allow more novelty. If the agent keeps trying variations of the same approach, raise it to 0.95.
