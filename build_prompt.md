# Build Prompt: Autonomous Coordinator–Subagent Research & Optimization Agent

## Role and Objective

You are an expert systems engineer building a **fully autonomous research agent** that continuously improves a target software project. The system runs an **infinite hypothesis→decompose→execute→integrate→test→learn loop** that **never terminates on its own** — it halts only on an explicit user stop signal (`Ctrl+C`, a `POST /stop` call, or a dashboard STOP button). Between start and stop it runs indefinitely without unbounded memory growth, context-window overflow, or hallucination drift.

Build in **Python 3.11+** using **asyncio**, backed by **Ollama** for all model inference and embeddings. Deliver production-quality, typed, testable code with clear module boundaries.

---

## System Overview

The agent automates the scientific method over a codebase. In each loop iteration a **coordinator agent** forms **exactly one hypothesis** about how to improve the target (informed by memory of past wins and mistakes). It **decomposes that single hypothesis** into independent subtasks and dispatches them to **parallel subagents**. The subagents are a **fan-out for speed, not a search over alternatives** — they all serve the *same* hypothesis, each doing a slice of the work. Subagents run **independently** (no shared live context, like Claude Code `Task` subagents), and each returns its result to the coordinator. The coordinator **reviews, edits, and integrates** their outputs into one combined change, runs the **opaque `test` oracle exactly once**, records the outcome (win or mistake) in **persistent memory**, and — if the score improved — **commits and pushes** to GitHub. It then forms the **next** hypothesis, refined from the accumulated memory, and repeats forever.

This is a **linear chain of hypotheses** (one at a time), with **map-reduce parallelism inside each iteration** (coordinator = reduce/integrate; subagents = map). There is **no** competing-branch search, bandit, or state tree.

**Heterogeneous models by role and skill.** The system is not tied to one model. The user declares in config which Ollama model serves as the **coordinator**, and a registry of **skill-specialized worker models** (e.g. a math-tuned model, a code-tuned model, a reasoning model), each tagged with the skills it is good at. When the coordinator decomposes a hypothesis, it assigns each subtask the **best-matching model by skill**, so a math-heavy subtask runs on the math model and a refactoring subtask runs on the code model — all within the same iteration. If a subtask has no matching specialist, or the user did not configure skill routing, the subagent falls back to a **mandatory default model** specified in config.

The user must observe everything live: the current hypothesis, the coordinator's decomposition, each subagent's assigned subtask and progress, the integration/review step, the single test score, memory writes, and GitHub saves.

---

## Core Requirements (all mandatory)

1. **Never self-terminates.** The loop condition is `while not stop_requested`. No convergence exit, no max-iteration exit, no "done" state. Budgets only throttle or pause. Cooperative shutdown drains in-flight subagents, flushes memory, exits cleanly.
2. **One hypothesis per loop.** Within a single iteration the hypothesis is fixed. All subagents work toward that one hypothesis. The next hypothesis is formed only after the current one is integrated, tested, and recorded.
3. **Coordinator + independent subagents.** A single coordinator owns hypothesis formation, decomposition, integration/review, testing, memory, and save. Subagents execute assigned subtasks in parallel and **do not share live context** with each other; they return results to the coordinator, which is the sole integration authority.
4. **Adaptive decomposition.** The coordinator decides *per hypothesis* whether to (a) assign explicit file/module scopes to each subagent, or (b) split into task-shaped subtasks — whichever fits the hypothesis. It also decides how many subagents to spawn (1..`max_subagents`); a trivial hypothesis may use one.
5. **Per-role and per-skill model selection.** The coordinator model, and a set of skill-tagged worker models, are all user-configurable. The coordinator routes each subtask to the worker model whose declared skills best match the subtask; unmatched subtasks and any unconfigured routing fall back to a **required default model**. Every model is an Ollama model name; the same infrastructure serves all of them. The chosen model for each subtask must be surfaced in the event stream and dashboard.
6. **Persistent memory** across three layers (below), surviving restarts, preventing repeated work and repeated mistakes.
7. **RAG retrieval** grounds hypothesis formation and every agent turn in retrieved facts + current file state, not accumulated transcript.
8. **Opaque `test` tool**: returns only `{score: float, remark: str | null}`. The agent never reads test source/data/internals. Enforced at the process boundary. Run **once per loop**, after integration.
9. **User-defined tools**: users register plain Python functions; schemas auto-generate for Ollama tool-calling. Reserved tool *kinds*: `test`, `save`, generic `action`.
10. **Bounded context** for both coordinator and each subagent: context is reassembled deterministically each turn from external memory under a hard token budget. Raw history is never accumulated. Primary anti-hallucination mechanism.
11. **Full observability**: structured event stream drives a live dashboard; every step inspectable; run pausable/resumable.
12. **GitHub `save`**: commit the integrated diff and push, **only** when the score strictly improves over the current baseline and the diff passes validation.

---

## Architecture — Modules

### 1. Coordinator (`coordinator/`)
The single control agent, running the **coordinator model** from config (hypothesis formation, decomposition, routing decisions, review/integration all use this model). Owns the infinite loop and all reduce-side logic. Per iteration:

1. **Form one hypothesis.** Assemble context (task spec + retrieved memory: relevant past wins and, explicitly, past *failures* labeled "already tried, failed because …"). Prompt Ollama for a single concrete, testable improvement + rationale.
2. **Anti-repetition gate.** Embed the hypothesis; query semantic memory. If cosine similarity to a prior *failed* hypothesis exceeds `dup_threshold`, reject and re-form (bounded retries), then force novelty (raise temperature / inject "avoid these approaches").
3. **Decompose (adaptive) + route models.** Choose scope-based or task-based split and a subagent count `n ∈ [1, max_subagents]`. Produce `n` **self-contained subtask briefs**, each with: goal, assigned file/module scope, constraints, expected output format, and a **required-skill tag(s)** describing what kind of expertise the subtask needs (e.g. `math`, `code`, `refactor`, `docs`, `reasoning`). Briefs must be independent (no cross-subagent dependencies), since subagents don't communicate. The coordinator then calls the **model router** (§Model Registry & Routing) to bind each brief to the Ollama model whose declared skills best match the brief's required-skill tags, falling back to the configured default model when there is no match.
4. **Dispatch** subagents concurrently (`asyncio.gather` under a semaphore), each in its **own git worktree** off the current baseline commit, **each running its routed model**.
5. **Review + integrate.** Collect each subagent's proposed edits/diff. The coordinator **reviews and may edit** each result (reject, patch, or accept), then **merges all accepted edits into one integration worktree**, resolving any overlap. Integration is a first-class LLM step, not a blind `git merge`.
6. **Test once.** Run the opaque `test` tool on the integration worktree → `(score, remark)`.
7. **Record.** Write the hypothesis, integrated diff, per-subagent contributions, score, remark, and outcome to memory — **wins and mistakes both**.
8. **Save on improvement.** If `score > baseline` and the diff passes validation, call `save` (commit + push), advance baseline, and fast-forward the target's working commit to the integrated result.
9. **Loop** — form the next hypothesis. Unconditionally, until `stop_requested`.

Loop skeleton (implement fully):
```python
async def run(self):
    while not self.stop_requested:
        await self.pause_gate.wait()                      # blocks when paused, never exits
        hyp = await self.form_hypothesis(self.memory)     # exactly one
        if self.memory.is_duplicate_failure(hyp):
            hyp = await self.reform_with_novelty(hyp)
        briefs = await self.decompose(hyp)                # 1..max_subagents, adaptive; each brief carries required skills
        for b in briefs:
            b.model = self.router.select(b.required_skills)  # skill->model, else default
        async with self.sem:
            results = await asyncio.gather(
                *[self.run_subagent(b) for b in briefs],   # each subagent runs b.model
                return_exceptions=True,                    # a crashed subagent never kills the loop
            )
        integrated = await self.review_and_integrate(hyp, self._ok(results))
        score, remark = await self.tools.call("test", workspace=integrated.path)
        self.memory.record(hyp, integrated, results, score, remark)
        if score > self.baseline and self.validate_diff(integrated):
            await self.tools.call("save", diff=integrated.diff, meta={"hyp": hyp, "score": score})
            self.baseline = score
            self.advance_working_commit(integrated)
        self.cleanup_worktrees(results, integrated)
        # loop continues unconditionally
    await self._drain_and_flush()
```

### 2. Subagent (`subagent/`)
A self-contained ReAct executor for **one subtask brief**, running the **model bound to that brief by the router** (`brief.model`), not necessarily the coordinator's model. It:
1. Creates a git worktree off the baseline commit (isolated; no contact with sibling subagents).
2. Assembles its **own** bounded context: the subtask brief, tool schemas, the file slices in its assigned scope (read fresh from disk), and any narrowly-relevant retrieved memory. **Not** other subagents' work, **not** the global transcript.
3. Runs an agent loop issuing tool calls (read/edit files in scope, user `action` tools) to complete the subtask, under a per-subagent step cap with rolling-summary compression every N steps.
4. Returns a structured result to the coordinator: `{subtask_id, diff, files_touched, summary, status}`. It **never** calls `test` or `save` (coordinator-only) and never merges.

### 3. Memory (`memory/`)
Three layers, distinct access patterns:
- **Episodic (SQLite)** — append-only ground truth. One row per iteration: `hypothesis, integrated_diff_hash, subagent_contribs, score, remark, outcome, baseline_before, ts`. Enables resume and the anti-repetition gate.
- **Semantic (LanceDB)** — embeddings of `hypothesis + outcome` for RAG retrieval and duplicate detection (Ollama `nomic-embed-text`). Store failures explicitly with remarks as negative examples.
- **Baseline/state (SQLite)** — current baseline score, current working commit, iteration counter; persisted for exact resume.

API: `record(...)`, `retrieve(query, k, include_failures=True)`, `is_duplicate_failure(hyp) -> bool`, `top_failures(context, k)`, `load_state()`, `save_state()`.

### 4. Tool Runtime (`tools/`)
- `@tool(name, description, kind="action"|"test"|"save")`; introspect signature + type hints → Ollama-compatible JSON schema.
- All tools execute in a **sandboxed subprocess** with time, memory, and default-deny network limits.
- **`test`**: runs in a **clean checkout of the integrated diff**, separate process; only stdout matching strict `{score, remark}` is returned. Agent context never contains test source/data. Tamper checks: reject if the diff modified the test harness or held-out assets (reward-hacking guard). **Coordinator-only, once per loop.**
- **`save`**: commit integrated diff to `github_branch_prefix/<iter>`, push to remote, return commit URL. **Coordinator-only, on validated improvement.**
- **User `action` tools**: auto-discovered from `user_tools/` at startup; callable by subagents and coordinator.

### 5. Model Registry & Routing (`models/`)
The single source of truth for which Ollama model does what. Loaded from config at startup.

- **Registry.** Parse config into: a **coordinator model** (required), a **default model** (required — the mandatory fallback), and a list of **worker models**, each an entry of `{name, skills: [str], (optional) options}` where `name` is an Ollama model name, `skills` are free-form capability tags the coordinator matches against (e.g. `math`, `code`, `refactor`, `docs`, `reasoning`, `test-analysis`), and `options` are per-model inference params (temperature, num_ctx, etc.).
- **Validation at startup.** Assert coordinator and default are present; verify every referenced model exists in Ollama (`GET /api/tags`) and fail fast with a clear message if one is missing or not pulled. Warn on duplicate skill tags.
- **`select(required_skills: list[str]) -> ModelSpec`.** Deterministic routing: score each worker model by overlap between its `skills` and the brief's `required_skills` (e.g. count of matched tags, tie-broken by registry order); return the best match. If no worker has any overlap, or `required_skills` is empty, or no worker models are configured, **return the default model**. Never raise — always resolve to a runnable model.
- **Client abstraction.** A thin async Ollama client keyed by model name so the coordinator and each subagent call `client.chat(model=spec.name, ...)` uniformly; per-model `options` from the registry are applied automatically. Embeddings always use the configured `embed_model` regardless of routing.
- Routing decisions (`brief -> chosen model + matched skills`) are emitted as events for the dashboard.

### 6. Observability (`server/`, `dashboard/`)
- FastAPI: `POST /start|/stop|/pause|/resume`, `GET /state`, `WebSocket /events`.
- Structured events: `hypothesis_formed, dup_rejected, decomposed, model_routed, subagent_spawned, subagent_progress, subagent_done, review_integrate, test_scored, memory_recorded, saved, paused, resumed, shutdown`. The `model_routed` and `subagent_spawned` events must include the chosen model name and matched skills.
- Minimal React dashboard: a header showing the **current hypothesis**; a **decomposition view** with one live panel per subagent showing its assigned scope, **the model it is running and why (matched skills, or "default fallback")**, progress, and returned diff; an **integration/review** panel; the single **test score** with remark and a score-over-iterations chart; a memory/timeline log; and a prominent **STOP** button (the only way the loop ends).

---

## Context Assembly (anti-hallucination — implement precisely)

Both coordinator and subagents reconstruct context as a **working set every turn**, never an accumulating transcript, under a hard token budget with fixed section allocations; on overflow, summarize or drop lowest-priority items.

**Coordinator context** (hypothesis formation + integration):
- ~20% task spec + tool schemas (fixed).
- ~30% retrieved memory: relevant past wins + explicitly labeled past failures.
- ~35% current baseline file slices relevant to the hypothesis (fresh from disk) + subagent result summaries during integration.
- ~15% rolling summary of the current iteration.

**Subagent context** (single subtask):
- ~20% subtask brief + tool schemas.
- ~50% in-scope file slices, fresh from disk each turn (re-anchor on real state).
- ~15% narrowly-relevant retrieved memory.
- ~15% rolling self-summary, regenerated every N steps.

Never include: sibling subagents' logs, full episodic history, unbounded ReAct scratchpad. Provide `assemble_context(role, inputs, memory, budget) -> str` and unit-test that output stays under budget across 100+ simulated turns.

---

## Reward-Hacking Safeguards (mandatory)

Optimizing an opaque score invites degenerate solutions. Enforce: test runs on a clean checkout of the integrated diff; the diff must touch relevant source and must not touch the test harness or held-out data (reject otherwise); optionally re-run the winner on a held-out invocation before `save`. Log every rejected/suspicious result to the event stream.

---

## Configuration (`config.yaml`)

Model selection is fully config-driven. Expose a `models` block plus the operational keys:

```yaml
ollama_host: "http://localhost:11434"

models:
  coordinator: "qwen2.5:14b-instruct"   # required: drives hypothesis/decompose/integrate
  default:     "qwen2.5-coder:7b"        # required: fallback for any unmatched subtask
  embed:       "nomic-embed-text"        # embeddings (routing never applies here)
  workers:                               # optional: skill-specialized models
    - name: "mathstral:7b"
      skills: ["math", "proof", "numeric"]
      options: { temperature: 0.2 }
    - name: "qwen2.5-coder:7b"
      skills: ["code", "refactor", "debug"]
    - name: "deepseek-r1:8b"
      skills: ["reasoning", "planning", "analysis"]
```

Routing rules the implementation must honor: `coordinator` and `default` are **required** (fail fast if absent); `workers` is optional — if omitted or empty, **every** subagent uses `default`. A subtask is routed to the worker with the greatest skill-tag overlap; ties break by list order; **no overlap → `default`**. All names must be pulled in Ollama (validate at startup). Per-model `options` override defaults for that model only.

Also expose: `max_subagents`, `context_token_budget`, section ratios (coordinator + subagent), `dup_threshold`, `subagent_step_cap`, `summary_every_n`, `target_repo`, `worktree_root`, `github_remote`, `github_branch_prefix`, sandbox limits, and `novelty_boost` (temperature bump when the hypothesis chain stagnates). **No key may cause termination** — budgets throttle/pause or nudge novelty only.

---

## Deliverables

1. Complete, typed, documented codebase with the module layout above.
2. `config.yaml` with sane defaults, a populated `models` block (coordinator + default + at least two skill-tagged workers), and an example `user_tools/` (a sample opaque `test`, the built-in `save`, one sample `action`).
3. Unit tests for: context-budget invariance (both roles), anti-repetition gate, memory persistence/resume, subagent worktree isolation, integration/merge of overlapping edits, the reward-hacking diff validator, and **model routing** (exact-match → specialist, no-match → default, empty-workers → default, missing coordinator/default → startup failure).
4. `setup.md` — full install and run instructions.
5. `ARCHITECTURE.md` — the loop, the never-stop guarantee, the coordinator/subagent (map-reduce) split, and how context stays bounded.

## Build Order (follow strictly)

1. **Single-agent loop, one subagent (n=1).** Coordinator forms a hypothesis, one subagent does all the work, coordinator integrates trivially, tests once, records, saves. Prove over 50+ iterations that memory prevents repetition and both context budgets hold with no drift, and that the loop never stops by itself.
2. **Fan-out + model routing.** Add adaptive decomposition, the model registry/router, and parallel independent subagents with per-worktree isolation — each subagent running its skill-routed model (default fallback until specialists are configured).
3. **Integration/review.** Add the coordinator's review-edit-merge step for overlapping/conflicting edits.
4. **Safeguards + dashboard.** Reward-hacking validation, then the live UI.

The single-agent grounded loop is the bulk of the risk; fan-out, integration, and UI are mechanical by comparison.

## Quality Bar

`mypy` clean, `ruff` formatted, async-correct (no blocking calls in the loop; offload inference/subprocess), fault-tolerant (a crashed subagent is caught, recorded as a mistake, and the loop continues), and resumable (`kill -9` then restart reconstructs baseline + memory + working commit from disk and continues the hypothesis chain).