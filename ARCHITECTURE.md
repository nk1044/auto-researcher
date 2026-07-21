# Architecture

## The Loop — Never Self-Terminates

The system runs one infinite loop controlled entirely by `Coordinator.run()`:

```python
async def run(self):
    while not self.stop_requested:          # only Ctrl+C / POST /stop sets this
        await self.pause_gate.wait()        # blocks on pause; never exits the loop
        await self._run_one_iteration()     # one full hypothesis cycle
    await self._drain_and_flush()           # clean shutdown
```

There is no convergence condition, no max-iteration limit, and no "done" state. The loop stops only when `stop_requested` is set via `Coordinator.stop()` (called by `POST /stop` or the dashboard Stop button). Token and step-cap budgets throttle behaviour — they never cause exit.

## One Hypothesis Per Iteration

Each call to `_run_one_iteration()` follows a fixed linear chain:

```
form_hypothesis → [anti-dup gate] → decompose + route → dispatch subagents
    → review+integrate → test once → record → [save if improved] → loop
```

The hypothesis is fixed for the entire iteration. All subagents serve the same hypothesis — they are a **speed fan-out** (map), not a search over alternatives. The coordinator is the sole **reduce/integrate** authority.

## Coordinator + Independent Subagents (Map-Reduce)

```
Coordinator  ── forms hypothesis ───────────────────────────────────► memory
               │
               ├── decompose → n briefs (1..max_subagents)
               │     each brief: goal, scope, skills, model
               │
               ├── dispatch ──► Subagent-1 (worktree-A, model-X)  ──┐
               │               ► Subagent-2 (worktree-B, model-Y)  ──┤ parallel
               │               ► Subagent-n (worktree-N, model-Z)  ──┘
               │                                                      │
               ├── collect results ◄───────────────────────────────────┘
               │
               ├── review+integrate → one unified diff (integration worktree)
               │
               ├── test once → (score, remark)
               │
               └── record + [save]
```

**Subagents are fully isolated**: each runs in its own git worktree off the same baseline commit, has no shared live context with siblings, and never calls `test` or `save` (coordinator-only). They return `SubtaskResult(diff, files_touched, summary, status)`.

## Context Stays Bounded

No raw transcript is ever accumulated. Both the coordinator and each subagent reconstruct their full context from scratch every turn, under a hard token budget (`context_token_budget`) with fixed section ratios.

**Coordinator context** (hypothesis formation + integration):

| Section | Ratio | Content |
|---------|-------|---------|
| Task spec + tool schemas | 20% | Fixed |
| Memory (wins + labeled failures) | 30% | Retrieved from semantic store |
| Current file slices | 35% | Fresh from disk each turn |
| Rolling iteration summary | 15% | Regenerated every N steps |

**Subagent context** (per ReAct step):

| Section | Ratio | Content |
|---------|-------|---------|
| Subtask brief + tools | 20% | Fixed brief |
| In-scope file slices | 50% | Fresh from disk |
| Retrieved memory | 15% | Narrow query |
| Rolling self-summary | 15% | Regenerated every N steps |

On overflow, sections are truncated proportionally — nothing is silently dropped. The `assemble_*_context` functions guarantee total output ≤ `token_budget × CHARS_PER_TOKEN`.

## Memory — Three Layers

```
Episodic (SQLite)   — append-only log. One row per iteration: hypothesis, diff hash,
                       score, remark, outcome, baseline_before, timestamp.
                       Powers resume: restart reads the last row.

Semantic (LanceDB)  — embeddings of "hypothesis + outcome" for RAG retrieval
                       and duplicate detection (nomic-embed-text, 768-dim vectors).
                       Failures are stored with explicit remarks as negative examples.
                       is_duplicate_failure() rejects hypotheses with cosine similarity
                       above dup_threshold (default 0.97) to any prior failure.

State (SQLite)      — single row: iteration counter, baseline score, working commit.
                       Written after every iteration. Exact resume on restart.
```

## Skill-Based Model Routing

The coordinator decides per subtask which Ollama model runs it:

1. Each subtask brief carries `required_skills: list[str]` (e.g. `["code", "refactor"]`).
2. `ModelRouter.select(required_skills)` scores each configured worker by skill overlap (intersection count). First-in-config wins ties.
3. No overlap → falls back to the mandatory `default` model.
4. The chosen model is set on `brief.model` before the subagent is dispatched.

All routing decisions are emitted as `model_routed` events, visible in the dashboard per subagent panel.

## Reward-Hacking Safeguards

The test oracle is opaque — the agent sees only `{score: float, remark: str}`, never its source, data, or internals. Additional guards:

1. **Clean checkout**: `test` runs on the integration worktree, not in-place.
2. **Diff validation** (`tools/validator.py`): before `save`, the diff must:
   - Be non-empty and touch at least one file.
   - Not touch any file matching `protected_patterns` (test harness, held-out data).
   - Contain valid `@@ ... @@` hunk headers.
3. **Rejection logging**: any rejected diff emits a `reward_hack_rejected` event with the reason, visible in the dashboard and event log.

## Startup Validation

`ModelRegistry.validate()` queries `GET /api/tags` on Ollama and fails fast if any referenced model is not pulled. The server will not start with a missing model. Missing `coordinator` or `default` keys in config raise `ValueError` immediately at registry construction.

## Shutdown

`Coordinator.stop()` sets `stop_requested = True` and unblocks `pause_gate` so a paused loop can exit. The current iteration finishes cleanly (in-flight subagents drain via `asyncio.gather(return_exceptions=True)`), memory is flushed with `save_state()`, and a `shutdown` event is emitted.

## Component Map

```
main.py
  └── server/app.py          — FastAPI + WebSocket, mounts EventBus
        └── coordinator/coordinator.py
              ├── coordinator/context.py       — bounded context assembly
              ├── coordinator/decomposer.py    — subtask decomposition
              ├── coordinator/integrator.py    — diff integration
              ├── models/registry.py           — startup validation
              ├── models/router.py             — skill-based routing
              ├── models/client.py             — async Ollama HTTP client
              ├── memory/__init__.py           — Memory façade
              │     ├── memory/episodic.py
              │     ├── memory/semantic.py
              │     └── memory/state.py
              └── subagent/subagent.py         — ReAct loop per worktree
                    ├── subagent/context.py
                    └── tools/
                          ├── decorator.py     — @tool + schema generation
                          ├── runtime.py       — sandboxed subprocess execution
                          ├── validator.py     — diff guard
                          └── save_tool.py     — git commit + push
```
