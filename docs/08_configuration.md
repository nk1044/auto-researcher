# Chapter 8 — Configuration (`config.yaml`)

## Overview

All runtime behaviour is driven by `config.yaml`. No behaviour is hard-coded that could reasonably be a config option. This chapter explains every key, its type, its default, and what happens if you get it wrong.

---

## Full Reference

```yaml
# ─── Ollama ───────────────────────────────────────────────────────────────────
ollama_host: "http://localhost:11434"

# ─── Models ───────────────────────────────────────────────────────────────────
models:
  coordinator: "qwen2.5:14b-instruct"
  default:     "qwen2.5-coder:7b"
  embed:       "nomic-embed-text"
  workers:
    - name:    "mathstral:7b"
      skills:  ["math", "proof", "numeric"]
      options:
        temperature: 0.2
    - name:    "qwen2.5-coder:7b"
      skills:  ["code", "refactor", "debug"]
    - name:    "deepseek-r1:8b"
      skills:  ["reasoning", "planning", "analysis"]

# ─── Target Repo ──────────────────────────────────────────────────────────────
target_repo:      "/path/to/your/repo"
worktree_root:    "/tmp/auto-researcher/worktrees"

# ─── Loop Control ─────────────────────────────────────────────────────────────
max_subagents:         4
subagent_step_cap:    20
summary_every_n:       5

# ─── Context ──────────────────────────────────────────────────────────────────
context_token_budget: 8192

# ─── Memory ───────────────────────────────────────────────────────────────────
data_dir:         "./data"
dup_threshold:    0.92
novelty_boost:    0.3

# ─── Save ─────────────────────────────────────────────────────────────────────
github_remote:          "origin"
github_branch_prefix:   "auto-researcher"
protected_patterns:
  - "tests/"
  - "test/"
  - "held_out/"
  - "eval/"
  - "benchmark/"
  - "*.test.*"
  - "test_*.py"
  - "*_test.py"

# ─── Server ───────────────────────────────────────────────────────────────────
host: "0.0.0.0"
port: 8000
```

---

## Section-by-Section

### Ollama

**`ollama_host`** (string, default `"http://localhost:11434"`)

The base URL of your Ollama server. If you are running Ollama on the same machine, the default works. If Ollama is on a different host (e.g. a GPU server), set this to `"http://<hostname>:<port>"`.

This value is used by three components:
- `ModelRegistry.validate()` — startup check for available models.
- `OllamaClient` — all chat and embedding requests.
- `SemanticMemory` — embedding requests for LanceDB.

---

### Models

**`models.coordinator`** (string, **required**)

The model used for all coordinator-level reasoning: hypothesis formation, decomposition, and integration review. Needs strong instruction-following and reliable JSON output. Larger models (14B+) generally produce better decompositions.

If this key is missing, `main.py` raises `ValueError` at startup.

**`models.default`** (string, **required**)

The fallback model for any subtask that has no matching specialist worker. This model must always be available. It is used when no worker's skills overlap with a subtask's `required_skills`, or when no workers are configured.

If this key is missing, `main.py` raises `ValueError` at startup.

**`models.embed`** (string, default `"nomic-embed-text"`)

The embedding model. All text embeddings in the system — for semantic memory storage, retrieval, and duplicate detection — use this model. `nomic-embed-text` produces 768-dimensional vectors. If you change this to a model that produces different-dimensioned vectors (e.g. 1024-dim), you **must** delete `data/semantic_db/` and let it rebuild. The LanceDB schema has the dimension hard-coded.

**`models.workers`** (list, optional)

A list of specialist models. Each worker has:
- **`name`** (string): Ollama model name. Must be listed in `ollama list`.
- **`skills`** (list of strings): free-form tags. The coordinator's decomposer assigns these to subtasks.
- **`options`** (dict, optional): per-model Ollama inference options. Common keys: `temperature`, `num_predict`, `top_p`, `top_k`, `repeat_penalty`.

Workers with no skill overlap with a subtask's `required_skills` are never selected for that subtask. If you remove all workers, the default model handles everything.

---

### Target Repo

**`target_repo`** (string, **required**)

Absolute path to the Git repository the agent will try to improve. Must be an existing directory with a `.git` folder. `main.py` checks this at startup and raises `ValueError` if the path does not exist or is not a Git repo.

**`worktree_root`** (string, default `"/tmp/auto-researcher/worktrees"`)

The directory where Git worktrees are created. Each subagent and the integration step each create one worktree here per iteration. They are deleted at the end of each iteration. The directory is created automatically if it does not exist.

Make sure this path is on the same filesystem as `target_repo` — Git worktrees do not work well across filesystems. If your repo is on a network mount, set `worktree_root` to a path on the same mount.

---

### Loop Control

**`max_subagents`** (integer, default `4`)

The maximum number of subagents that can run concurrently. Implemented as an `asyncio.Semaphore`. With a 4-core machine and slow models, 4 is a reasonable default. On a machine with a large GPU that can multiplex inference, you can increase this.

The coordinator's decomposer also uses this as the upper bound for the number of subtasks it produces per iteration. It will never decompose into more subtasks than this value.

**`subagent_step_cap`** (integer, default `20`)

The maximum number of ReAct loop steps a subagent can take before it is cut off with `PARTIAL` status. If you see many `PARTIAL` results in the dashboard, increase this. If subagents are spending too long on simple tasks (model gets confused), decrease it.

Rule of thumb: a typical code modification task takes 3–7 steps. 20 is generous.

**`summary_every_n`** (integer, default `5`)

How often (in steps) the subagent regenerates its rolling summary of progress. Lower values mean the model has a more up-to-date summary in context, at the cost of more LLM calls. The minimum useful value is 3.

---

### Context

**`context_token_budget`** (integer, default `8192`)

The maximum number of tokens allocated to the context assembled for each model call. The context assembly code converts this to a character budget using `CHARS_PER_TOKEN = 3` (a conservative approximation for code-heavy content, so 8192 tokens → 24,576 characters).

This budget applies to **both** subagent context (20/50/15/15 split) and coordinator context (20/30/35/15 split). The budget is a hard cap — content is truncated, not dropped.

Set this to roughly `model_context_window * 0.6` to leave room for the system prompt and the model's response. For models with an 8K context window, 8192 is appropriate. For models with a 32K window, you could safely go to 16384 or higher.

---

### Memory

**`data_dir`** (string, default `"./data"`)

The directory where all persistent storage lives. Created automatically if it does not exist. Contains:
- `episodic.db` — SQLite database of all iteration records.
- `state.db` — SQLite database with the current agent state (one row).
- `semantic_db/` — LanceDB directory with vector embeddings.

To start completely fresh (forgetting all past iterations): `rm -rf ./data`.

To start fresh while keeping episodic history for auditing: `rm -rf ./data/semantic_db ./data/state.db`.

**`dup_threshold`** (float, default `0.92`)

The cosine similarity threshold above which a new hypothesis is considered a duplicate of a past failure. Range: 0.0 (never duplicate) to 1.0 (exact duplicate only).

Tuning guide:
- Too many `dup_rejected` events → lower the threshold (0.85–0.88).
- Agent keeps re-trying similar approaches → raise the threshold (0.95–0.98).
- A good balance for diverse coding problems is around 0.90–0.93.

**`novelty_boost`** (float, default `0.3`)

How much to add to the coordinator model's temperature when reforming a rejected hypothesis. The base temperature comes from `models.coordinator.options.temperature` (or 0.7 if not set). A boost of 0.3 means `temperature = base + 0.3`.

Higher boost = more creative (and potentially less coherent) reformulation. If the reformed hypotheses are nonsensical, reduce this to 0.1–0.2.

---

### Save

**`github_remote`** (string, default `"origin"`)

The Git remote to push improvements to. Run `git remote -v` in your target repo to see available remotes. For GitHub repos, `"origin"` is almost always correct.

**`github_branch_prefix`** (string, default `"auto-researcher"`)

The prefix for saved branches. Iteration 42 gets pushed to branch `auto-researcher/0042`. If you run multiple experiments on the same repo, change this to namespace them: `experiment-refactor`, `experiment-tests`, etc.

**`protected_patterns`** (list of strings)

File patterns that are never allowed in a saved diff. If a diff touches any path matching these patterns, it is rejected with `REWARD_HACK_REJECTED` before saving.

Pattern matching uses Python's `fnmatch` plus prefix matching:
- `"tests/"` matches any file under a `tests/` directory at any depth.
- `"test_*.py"` matches files named `test_foo.py`.
- `"*.test.*"` matches `foo.test.js`, `bar.test.ts`, etc.

Add your own patterns if you have additional evaluation or benchmark files you want to protect. The more protected paths you add, the safer the system is against reward hacking — but the harder it is for the agent to add new test coverage (which is usually legitimate).

---

### Server

**`host`** (string, default `"0.0.0.0"`)

The interface the FastAPI server binds to. `0.0.0.0` means all interfaces (accessible from other machines on the network). Use `127.0.0.1` if you want to restrict access to localhost only.

**`port`** (integer, default `8000`)

The port the server listens on. Change this if port 8000 is already in use.

---

## Validation at Startup

`main.py` validates the following at startup, before the server starts:

| Check | Error if... |
|-------|------------|
| `target_repo` | Path does not exist |
| `target_repo` | Path has no `.git` directory |
| `models.coordinator` | Key missing |
| `models.default` | Key missing |
| All model names | Not in `ollama list` (validated by `ModelRegistry.validate()`) |

A validation failure prints a clear error message and exits with code 1. It never starts the server in a broken state.

---

## Environment Variable Overrides

Any config value can be overridden with an environment variable of the form `AR_<KEY_UPPERCASE>`. For example:

```bash
AR_TARGET_REPO=/path/to/other-repo python main.py
```

This is useful for CI or Docker deployments where you want a single `config.yaml` template but different values per environment. Nested keys use double underscores: `AR_MODELS__COORDINATOR=qwen2.5:7b`.

Environment variables take precedence over `config.yaml` values but are applied after file loading — the same validation runs regardless of source.

---

## Example: Minimal Config

```yaml
ollama_host: "http://localhost:11434"
models:
  coordinator: "qwen2.5:14b-instruct"
  default:     "qwen2.5-coder:7b"
  embed:       "nomic-embed-text"
target_repo: "/home/you/my-project"
```

Everything else uses defaults. This is sufficient to start the agent on a small project with a single model doing all the work.

---

## Example: Multi-Specialist Config

```yaml
ollama_host: "http://gpu-server:11434"
models:
  coordinator: "qwen2.5:32b-instruct"
  default:     "qwen2.5-coder:7b"
  embed:       "nomic-embed-text"
  workers:
    - name:   "qwen2.5-coder:14b"
      skills: ["code", "refactor", "debug", "optimization"]
      options: { temperature: 0.3 }
    - name:   "deepseek-r1:14b"
      skills: ["reasoning", "planning", "analysis", "architecture"]
      options: { temperature: 0.5 }
    - name:   "mathstral:7b"
      skills: ["math", "proof", "numeric", "algorithm"]
      options: { temperature: 0.1 }
target_repo: "/mnt/projects/my-repo"
worktree_root: "/mnt/projects/worktrees"
max_subagents: 8
subagent_step_cap: 30
context_token_budget: 16384
dup_threshold: 0.90
github_branch_prefix: "experiment-v2"
protected_patterns:
  - "tests/"
  - "benchmarks/"
  - "eval/"
  - "*.eval.*"
```
