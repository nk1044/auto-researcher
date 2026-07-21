# Chapter 0 — System Overview

## What Is This?

Auto-Researcher is an **autonomous software improvement agent**. You point it at a Git repository and it runs forever, proposing and testing one improvement at a time, learning from what works and what fails, and committing changes to GitHub only when the score strictly gets better.

It is not an AI chatbot. It is not a one-shot code-generation tool. It is a **running process** that you start, let work, and stop when you are satisfied — like a tireless junior engineer who never sleeps, never forgets a past mistake, and always reads the current code before touching it.

---

## The Core Idea: The Scientific Method Over Code

Every iteration of the loop is one experiment:

1. **Hypothesis** — "If I inline this loop, the tests will be faster."
2. **Experiment** — Fan out parallel workers to make the change.
3. **Test** — Run the opaque test oracle once on the result.
4. **Record** — Write the outcome (win or mistake) to memory.
5. **Learn** — Use that memory to form the next hypothesis.

The loop never terminates on its own. It stops only when you press **STOP** (or send `POST /stop`). There is no convergence exit, no "done" state, no max-iteration guard.

---

## The Loop in One Diagram

```
                        ┌────────────────────────────────────────────────┐
                        │            COORDINATOR (infinite loop)          │
                        │                                                  │
  Memory ─────────────► │  1. form_hypothesis()                           │
  (wins + failures)     │  2. anti-dup gate  ──► reject + reform          │
                        │  3. decompose()  ────► n SubtaskBriefs           │
                        │  4. route models per brief                       │
                        │  5. dispatch n subagents ──────────────┐        │
                        │                                        │        │
                        │         ┌──────────────┐   ┌──────────┴──────┐  │
                        │         │  Subagent-1  │   │   Subagent-2    │  │
                        │         │ (worktree A) │   │  (worktree B)   │  │
                        │         │  model X     │   │   model Y       │  │
                        │         └──────┬───────┘   └────────┬────────┘  │
                        │                └─────────┬──────────┘            │
                        │                          │ SubtaskResults        │
                        │  6. review_and_integrate()                      │
                        │  7. test once ────────── score, remark          │
                        │  8. memory.record()                              │
                        │  9. if score improved → save (git push)         │
                        │  10. loop ────────────────────────────────────► │
                        └────────────────────────────────────────────────┘
```

This is **map-reduce parallelism inside a linear chain**:
- The chain of hypotheses is strictly one-at-a-time (no branching, no search tree).
- Within each iteration, the subagents **map** (execute in parallel), and the coordinator **reduces** (integrates their results).

---

## Key Design Principles

### 1. Never Self-Terminates
The loop condition is `while not stop_requested`. There is literally no other exit. Budgets (token limits, step caps) throttle or nudge behaviour — they never cause the process to exit.

### 2. One Hypothesis Per Iteration
A hypothesis is formed, pursued, and resolved before the next one is considered. All subagents in an iteration work toward the same hypothesis. There is no competing-branch search.

### 3. Subagents Are Isolated
Each subagent gets its own Git worktree (a separate checkout of the same commit). It cannot read or write the work of its siblings. It returns a `diff` to the coordinator, which is the sole integration authority.

### 4. Context Is Bounded — Never Accumulating
Neither the coordinator nor any subagent keeps a growing chat transcript. Every turn, context is reconstructed from scratch under a hard token budget, pulling only what is needed from external memory and fresh file reads. This is the primary anti-hallucination mechanism.

### 5. The Test Oracle Is Opaque
The agent never sees the test source, test data, or how scoring works. It only receives `{score: float, remark: str}`. A diff validator enforces that the agent cannot modify the test harness (reward-hacking guard).

### 6. Heterogeneous Models
Different subtasks can run on different Ollama models. A math-heavy subtask can be routed to a math-tuned model; a refactoring subtask to a code-tuned model. All routing is deterministic and config-driven.

---

## Repository Layout

```
auto-researcher/
│
├── main.py                  ← entry point; wires everything together
├── config.yaml              ← all knobs (models, limits, paths)
├── requirements.txt
│
├── shared/
│   └── types.py             ← all shared dataclasses (read this first)
│
├── memory/
│   ├── episodic.py          ← SQLite: one row per iteration (append-only log)
│   ├── semantic.py          ← LanceDB: vector embeddings for RAG + dup detection
│   ├── state.py             ← SQLite: current baseline score, commit, iteration
│   └── __init__.py          ← Memory façade (unified API)
│
├── models/
│   ├── registry.py          ← parses config, validates models exist in Ollama
│   ├── router.py            ← skill-based routing: pick best model for a subtask
│   └── client.py            ← async Ollama HTTP client
│
├── tools/
│   ├── decorator.py         ← @tool decorator + JSON schema auto-generation
│   ├── runtime.py           ← tool registry + sandboxed subprocess execution
│   ├── validator.py         ← diff validator (reward-hacking guard)
│   └── save_tool.py         ← git commit + push on score improvement
│
├── subagent/
│   ├── subagent.py          ← ReAct loop executor (one subtask, one worktree)
│   └── context.py           ← bounded context assembly for subagents
│
├── coordinator/
│   ├── coordinator.py       ← THE infinite loop; hypothesis→decompose→...→record
│   ├── context.py           ← bounded context assembly for the coordinator
│   ├── decomposer.py        ← turns a hypothesis into n SubtaskBriefs
│   └── integrator.py        ← merges subagent diffs into one integration worktree
│
├── server/
│   ├── app.py               ← FastAPI: /start /stop /pause /resume /state /events
│   └── events.py            ← event types + async pub/sub EventBus
│
├── dashboard/
│   └── index.html           ← live dashboard (plain HTML/JS, no build step)
│
├── user_tools/
│   ├── test.py              ← sample test oracle (replace with your own)
│   └── sample_action.py     ← sample action tool available to subagents
│
└── tests/                   ← pytest test suite (no Ollama required)
```

---

## Reading Order for New Contributors

If you want to understand the codebase from the ground up:

1. `shared/types.py` — understand the data model first
2. `server/events.py` — understand how observability works
3. `memory/` — understand how the system remembers
4. `models/` — understand how model selection works
5. `tools/` — understand the tool system
6. `subagent/` — understand how one subtask is executed
7. `coordinator/coordinator.py` — the main loop (everything comes together here)
8. `server/app.py` + `dashboard/index.html` — the user-facing control interface

Each of the following chapters covers one module in depth.
