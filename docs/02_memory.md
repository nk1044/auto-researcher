# Chapter 2 — Memory System (`memory/`)

## Purpose

The memory system does three things:

1. **Prevents repetition** — avoids re-trying hypotheses that already failed.
2. **Grounds the coordinator** — provides relevant wins and failures to inform each new hypothesis.
3. **Enables resume** — stores enough state that a killed process can restart and continue from exactly where it left off.

Without memory, every iteration is a random walk. With memory, the hypothesis chain improves systematically.

---

## Three Layers, Three Files

The memory system uses three storage backends, each optimised for a different access pattern:

| Layer | File | Backend | Purpose |
|-------|------|---------|---------|
| Episodic | `memory/episodic.py` | SQLite | Append-only log of all iterations |
| Semantic | `memory/semantic.py` | LanceDB | Vector embeddings for RAG and dup detection |
| State | `memory/state.py` | SQLite | Current baseline, commit, iteration counter |

All three are exposed through a single façade class in `memory/__init__.py`.

---

## Layer 1: Episodic Memory (`memory/episodic.py`)

### What It Does

The episodic layer is a **permanent append-only log**. Every iteration — win or mistake — is written here. You can think of it as the system's journal.

### The Database Schema

```sql
CREATE TABLE iterations (
    id                   TEXT PRIMARY KEY,
    hypothesis           TEXT,
    integrated_diff_hash TEXT,     -- SHA256[:16] of the merged diff
    subagent_contribs    TEXT,     -- JSON array: [{id, status, files}]
    score                REAL,
    remark               TEXT,
    outcome              TEXT,     -- "win" | "mistake" | "neutral"
    baseline_before      REAL,
    ts                   TEXT,     -- ISO 8601 timestamp
    iteration            INTEGER
);
```

### Key Methods

```python
class EpisodicMemory:
    def record(self, record: IterationRecord) -> None
        # INSERT OR REPLACE: idempotent, safe to call twice

    def get_recent(self, k: int = 20) -> list[IterationRecord]
        # Returns last k rows, most-recent first

    def get_last(self) -> Optional[IterationRecord]
        # Returns the single most recent row (used for resume)

    def count(self) -> int
        # Total number of iterations recorded
```

### Thread Safety

The SQLite connection is opened with `check_same_thread=False` and all operations are guarded by a `threading.Lock`. This is necessary because the coordinator runs in an asyncio event loop but SQLite is synchronous — the `threading.Lock` prevents races when `asyncio.to_thread()` runs database operations in the thread pool.

### Why Append-Only?

Append-only design means you never lose history. Even if a hypothesis looked bad at the time but later turns out to be related to a breakthrough, the record exists. It also makes the table easy to reason about: you never need to worry about an UPDATE racing with an INSERT.

---

## Layer 2: Semantic Memory (`memory/semantic.py`)

### What It Does

The semantic layer stores **vector embeddings** of hypothesis text alongside the outcome (win or mistake). It enables two critical operations:

1. **RAG retrieval**: "Given this new hypothesis, what relevant past experiences should I know about?"
2. **Duplicate detection**: "Have I tried something too similar to a past failure?"

### The Technology Stack

- **LanceDB**: an embedded vector database (no server needed, stores to disk).
- **nomic-embed-text**: the embedding model (768-dimensional vectors, run via Ollama's `/api/embeddings` endpoint).
- **pyarrow**: used to define the table schema with a fixed-size vector column.

### The Table Schema

```python
schema = pa.schema([
    pa.field("id",        pa.string()),
    pa.field("text",      pa.string()),    # hypothesis text
    pa.field("outcome",   pa.string()),    # "win" | "mistake"
    pa.field("score",     pa.float32()),
    pa.field("remark",    pa.string()),
    pa.field("iteration", pa.int32()),
    pa.field("ts",        pa.string()),
    pa.field("vector",    pa.list_(pa.float32(), 768)),  # fixed-size
])
```

The `pa.list_(pa.float32(), 768)` creates a fixed-size list column — LanceDB requires this for approximate nearest-neighbor search to work efficiently.

### Key Methods

```python
class SemanticMemory:
    async def embed(self, text: str) -> list[float]
        # Calls POST /api/embeddings on the Ollama server
        # Returns a list of 768 floats

    async def store(self, entry: MemoryEntry) -> None
        # Embeds entry.text, upserts into LanceDB
        # Upsert = delete existing row by id, then add new row

    async def retrieve(
        self, query: str, k: int = 5, include_failures: bool = True
    ) -> list[MemoryEntry]
        # Embeds query, runs ANN search
        # include_failures=False filters to only WIN outcomes

    async def is_duplicate_failure(
        self, text: str, threshold: float = 0.92
    ) -> bool
        # Embeds text, searches only MISTAKE entries
        # Returns True if nearest failure has cosine similarity > threshold

    async def top_failures(
        self, context: str, k: int = 5
    ) -> list[MemoryEntry]
        # Like retrieve() but only returns MISTAKE entries
        # Used to populate the "already tried, avoid these" section of context
```

### How Cosine Similarity Works Here

LanceDB's `.search(vector)` returns rows with a `_distance` field. For cosine distance, `_distance` is in `[0, 2]` where 0 = identical. The code converts this to similarity: `cosine_sim = 1.0 - distance`. A similarity of 0.92 means the hypotheses are 92% semantically similar.

The `dup_threshold` (default 0.92, configurable) controls how aggressive the anti-repetition gate is. Set it lower (e.g. 0.80) to reject hypotheses that are only vaguely related to past failures; set it higher (e.g. 0.98) to only reject near-exact repeats.

### The asyncio Wrapping Pattern

LanceDB is a synchronous library. Every LanceDB call is wrapped in `asyncio.to_thread()`:

```python
rows = await asyncio.to_thread(_search)
```

This runs the synchronous database operation on the thread pool without blocking the asyncio event loop. The coordinator and subagents are fully async, so blocking the event loop even briefly would delay everything.

---

## Layer 3: State Memory (`memory/state.py`)

### What It Does

State memory stores exactly one row: the current agent state. It is written after every iteration and read on startup.

### The Database Schema

```sql
CREATE TABLE state (
    key   TEXT PRIMARY KEY,
    value TEXT    -- JSON-serialized AgentState
);
```

There is always at most one row, with `key = 'agent_state'`. This is the simplest possible persistent key-value store.

### Key Methods

```python
class StateMemory:
    def save_state(self, state: AgentState) -> None
        # Serializes AgentState to JSON, does INSERT OR REPLACE

    def load_state(self) -> Optional[AgentState]
        # Returns None if no state exists (first run)
        # Deserializes JSON back to AgentState
```

### The Resume Mechanism

When the coordinator starts, it calls `memory.load_state()`. If a state exists, it restores `iteration`, `baseline_score`, and `working_commit`. The loop then continues from that iteration number and continues branching from that commit — as if the process never stopped.

If `load_state()` returns `None` (first run), the coordinator calls `git rev-parse HEAD` on the target repo to get the starting commit and uses `baseline_score = 0.0`.

---

## The Memory Façade (`memory/__init__.py`)

The `Memory` class is the only thing the rest of the system imports. It wraps all three layers and provides a clean unified API.

```python
class Memory:
    def __init__(
        self,
        data_dir: str,
        ollama_host: str,
        embed_model: str,
        dup_threshold: float = 0.92,
    ) -> None
        # Creates data_dir if it doesn't exist
        # Instantiates all three layers with their respective paths

    async def init(self) -> None
        # Calls init_db() on episodic and state
        # Calls await semantic.init() to create the LanceDB table

    async def record(self, record: IterationRecord) -> None
        # Writes to BOTH episodic (sync) and semantic (async)
        # The semantic store embeds the hypothesis and indexes it

    async def retrieve(self, query: str, k: int = 5, include_failures: bool = True)
    async def is_duplicate_failure(self, hypothesis_text: str) -> bool
    async def top_failures(self, context: str, k: int = 5)

    def load_state(self) -> Optional[AgentState]
    def save_state(self, state: AgentState) -> None
    def get_recent_iterations(self, k: int = 20) -> list[IterationRecord]
    def iteration_count(self) -> int
```

### Data Directory Layout

```
data/
├── episodic.db          ← SQLite, table: iterations
├── state.db             ← SQLite, table: state
└── semantic_db/         ← LanceDB directory
    └── memories.lance/  ← LanceDB internal storage
```

All three stores live under the `data_dir` configured in `config.yaml` (default `./data`). To wipe the memory and start fresh, delete this directory.

---

## How Memory Shapes Hypothesis Formation

Here is the exact flow each iteration:

```
1. coordinator asks: memory.retrieve("improve test score", k=5, include_failures=False)
   → returns 5 past WINs most similar to "improve test score"
   → coordinator uses these as "what has worked before"

2. coordinator asks: memory.top_failures(current_hypothesis_text, k=5)
   → returns 5 past MISTAKEs most similar to what we're about to try
   → coordinator uses these as "what has already failed"

3. coordinator forms hypothesis, then asks: memory.is_duplicate_failure(hypothesis.text)
   → if True: reject, boost temperature, reform with novelty
   → if False: proceed to decompose
```

This three-step grounding ensures that every hypothesis is both informed by past success and actively avoiding past failure.

---

## For Contributors: Extending the Memory System

**Adding a new field to iteration records:**
1. Add the field to `IterationRecord` in `shared/types.py`.
2. Update `EpisodicMemory.init_db()` to add the column (add `IF NOT EXISTS` guards for backward compatibility).
3. Update `EpisodicMemory.record()` to write the new field.
4. Update `EpisodicMemory._row_to_record()` to read it back.

**Changing the embedding model:**
Edit `config.yaml` → `models.embed`. The new model must be pulled in Ollama. Note: if you change the embedding model, the vector dimensions may change — you will need to delete `data/semantic_db/` and let it rebuild (you will lose semantic similarity history, but episodic history is preserved).

**Using a different vector database:**
Replace `memory/semantic.py` entirely. The `Memory` façade only calls `store()`, `retrieve()`, `is_duplicate_failure()`, and `top_failures()` — implement those four methods and the rest of the system does not need to change.
