# Chapter 3 — Model Registry, Routing, and Client (`models/`)

## Overview

This module answers three questions:

1. **Which models are available?** (`registry.py`)
2. **Which model should run this subtask?** (`router.py`)
3. **How do we talk to Ollama?** (`client.py`)

The system is designed to be model-agnostic. Any Ollama-compatible model can be used for any role. The config file drives everything.

---

## Model Registry (`models/registry.py`)

### Purpose

The registry is the single authoritative source of which models exist, what skills they have, and what inference parameters they use. It is loaded once at startup and never changes while the process runs.

### Parsing Config

The registry reads from the `models:` block of `config.yaml`:

```yaml
models:
  coordinator: "qwen2.5:14b-instruct"
  default:     "qwen2.5-coder:7b"
  embed:       "nomic-embed-text"
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

- **`coordinator`** (required): the model that forms hypotheses, decomposes them, and integrates results. Needs strong instruction-following and JSON output.
- **`default`** (required): the fallback model for any subtask that has no matching specialist. Must always be available.
- **`embed`**: the model used for all embeddings. `nomic-embed-text` produces 768-dim vectors. This model is never routed — it is always the same.
- **`workers`**: optional list of specialists. Each has a `name` (Ollama model), `skills` (free-form tags), and optionally per-model `options`.

### Startup Validation

At startup, `registry.validate()` is called. It hits Ollama's `GET /api/tags` endpoint and checks that every model name referenced in the config is in the response. If any model is missing, it raises `ValueError` with a clear message listing the missing models and the `ollama pull <name>` command to fix it.

This fail-fast behaviour is intentional. A missing model discovered on the first hypothesis attempt would waste a full iteration. Better to crash immediately on startup.

It also scans worker skill tags for duplicates and emits a `logger.warning` — duplicate skill tags make routing ambiguous (both workers have the same skill, but only one wins by list order).

### Properties

```python
registry.coordinator  # → ModelSpec("qwen2.5:14b-instruct", skills=[], ...)
registry.default      # → ModelSpec("qwen2.5-coder:7b", skills=[], ...)
registry.embed_model  # → "nomic-embed-text"  (str, not ModelSpec)
registry.workers      # → [ModelSpec("mathstral:7b", skills=["math","proof","numeric"]), ...]
```

---

## Model Router (`models/router.py`)

### Purpose

Given a subtask's `required_skills`, deterministically pick the best matching worker model.

### The Algorithm

```python
def select(self, required_skills: list[str]) -> tuple[ModelSpec, list[str], bool]:
    # Returns (chosen_model, matched_skills, is_fallback)
```

**Step 1:** If `required_skills` is empty, or no workers are configured, return `(default, [], True)`.

**Step 2:** For each worker, count how many of its skills appear in `required_skills`:

```
subtask requires: ["code", "refactor", "debug"]

Worker A — skills: ["code", "refactor"]    → overlap = 2
Worker B — skills: ["math", "proof"]       → overlap = 0
Worker C — skills: ["code"]                → overlap = 1

Winner: Worker A (overlap = 2)
```

**Step 3:** The worker with the highest overlap wins. If two workers tie, the first one in the config wins (registry order).

**Step 4:** If no worker has any overlap (score = 0), return `(default, [], True)`.

**Why this is deterministic:** The score function is pure (no randomness), and ties are broken by a fixed ordering. The same config + same subtask always produces the same routing decision.

### The Return Value

The router always returns a 3-tuple:
- `ModelSpec` — the chosen model (never `None`, never raises)
- `list[str]` — which skills actually matched (empty for default fallback)
- `bool` — `True` if the default model was used (no match found)

This 3-tuple is immediately stored on the `SubtaskBrief` and emitted as a `model_routed` event. The dashboard shows each subagent's chosen model and whether it was a specialist or a fallback.

### Never Raises

`ModelRouter.select()` has no failure modes. If the registry has no workers, it returns the default. If the skills are nonsense, it returns the default. This is a deliberate design choice: a routing failure should never crash the loop.

---

## Ollama Client (`models/client.py`)

### Purpose

A thin async HTTP wrapper around the Ollama REST API. Every chat and embedding call in the system goes through this client.

### `ChatMessage` and `ChatResponse`

```python
@dataclass
class ChatMessage:
    role:         str           # "system" | "user" | "assistant" | "tool"
    content:      str
    tool_calls:   list[dict]    # present in assistant messages that call tools
    tool_call_id: str           # present in tool-result messages
```

```python
@dataclass
class ChatResponse:
    content:          str
    tool_calls:       list[dict]    # parsed from response.message.tool_calls
    model:            str
    prompt_tokens:    int           # from prompt_eval_count
    completion_tokens: int          # from eval_count
```

These dataclasses are the system's interface to Ollama. All coordinator and subagent logic works with `ChatMessage` objects; the raw Ollama JSON format is an implementation detail of `client.py`.

### `OllamaClient`

```python
class OllamaClient:
    def __init__(self, base_url: str, registry: ModelRegistry, timeout: float = 120.0)
        # Creates a single persistent httpx.AsyncClient
        # base_url is typically "http://localhost:11434"
```

**`chat(model_spec, messages, tools=None, options_override=None)`**

Sends a `POST /api/chat` request. Key behaviours:
- Always uses `stream: false` — the whole response comes back at once.
- Merges `model_spec.options` with `options_override` (override wins). This is how `novelty_boost` works: the coordinator passes `{"temperature": base + 0.3}` when reforming after a dup rejection.
- Tool schemas are passed as `tools` in the request body; Ollama returns `tool_calls` in the response when the model decides to call a tool.

**`embed(text, model=None)`**

Sends a `POST /api/embeddings` request. If `model` is `None`, uses `registry.embed_model`. Returns a `list[float]`.

**`list_models()`**

Sends `GET /api/tags`. Returns a flat `list[str]` of model names. Used by `ModelRegistry.validate()`.

**`registry` property**

The client exposes `registry` as a read-only property. The coordinator accesses `self.client.registry.coordinator` to get the coordinator's own `ModelSpec`.

### Connection Pooling

A single `httpx.AsyncClient` is created at startup and reused for all requests. This is important because Ollama can have latency on the first request to a model (loading it into GPU memory). Reusing the client keeps the TCP connection alive and reduces overhead.

The `timeout=120.0` default is generous — large models generating long responses can take 60+ seconds. This can be tuned in the config under per-model `options`.

---

## How It All Fits Together

Here is the sequence from coordinator hypothesis to subagent model selection:

```
1. Coordinator calls decompose(hyp)
   └─► LLM call: coordinator model → returns n subtask dicts with required_skills

2. Coordinator iterates over briefs:
   for brief in briefs:
       model_spec, matched, fallback = router.select(brief.required_skills)
       brief.model = model_spec
       brief.matched_skills = matched
       brief.fallback = fallback
       emit(MODEL_ROUTED, {model: model_spec.name, matched_skills: matched, fallback: fallback})

3. Coordinator dispatches subagents:
   Subagent(brief=brief, client=client, ...)
   └─► subagent.run() calls client.chat(model_spec=brief.model, messages=...)
```

Every subagent uses `brief.model` — the model assigned to it by the router. If you add a new worker to the config with skill `"sql"` and a subtask requires `"sql"`, it will automatically route to that worker without any code changes.

---

## For Contributors: Adding a New Worker Model

1. Pull the model: `ollama pull <name>`
2. Add it to `config.yaml`:
   ```yaml
   workers:
     - name: "your-model:tag"
       skills: ["your-skill", "another-skill"]
       options:
         temperature: 0.4
   ```
3. Restart the server. The registry validates on startup.

The coordinator's decomposer will now produce subtasks tagged with your skill, and the router will send them to your new model automatically.

## For Contributors: Replacing Ollama

The rest of the system only knows about `OllamaClient`. If you want to use a different inference backend (OpenAI-compatible API, vLLM, etc.), create a drop-in replacement for `OllamaClient` that implements the same interface: `chat()`, `embed()`, `list_models()`, `close()`. Replace the instantiation in `main.py` and nothing else needs to change.
