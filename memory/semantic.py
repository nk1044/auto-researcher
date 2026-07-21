from __future__ import annotations

import asyncio
from typing import Optional

import httpx

from shared.types import MemoryEntry, OutcomeType

_VECTOR_DIM = 768


def _import_lancedb():
    try:
        import lancedb
        return lancedb
    except ImportError as exc:
        raise ImportError(
            "lancedb is required for SemanticMemory. "
            "Install it with: pip install lancedb"
        ) from exc


def _import_pyarrow():
    try:
        import pyarrow as pa
        return pa
    except ImportError as exc:
        raise ImportError(
            "pyarrow is required for SemanticMemory. "
            "Install it with: pip install pyarrow"
        ) from exc


class SemanticMemory:
    def __init__(self, db_path: str, ollama_host: str, embed_model: str) -> None:
        self._db_path = db_path
        self._ollama_host = ollama_host.rstrip("/")
        self._embed_model = embed_model
        self._db = None
        self._table = None

    async def init(self) -> None:
        lancedb = _import_lancedb()
        pa = _import_pyarrow()

        schema = pa.schema(
            [
                pa.field("id", pa.string()),
                pa.field("text", pa.string()),
                pa.field("outcome", pa.string()),
                pa.field("score", pa.float32()),
                pa.field("remark", pa.string()),
                pa.field("iteration", pa.int32()),
                pa.field("ts", pa.string()),
                pa.field("vector", pa.list_(pa.float32(), _VECTOR_DIM)),
            ]
        )

        self._db = await asyncio.to_thread(lancedb.connect, self._db_path)

        existing = await asyncio.to_thread(self._db.table_names)
        if "memories" in existing:
            self._table = await asyncio.to_thread(self._db.open_table, "memories")
        else:
            self._table = await asyncio.to_thread(
                self._db.create_table, "memories", schema=schema
            )

    async def embed(self, text: str) -> list[float]:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"{self._ollama_host}/api/embeddings",
                json={"model": self._embed_model, "prompt": text},
            )
            response.raise_for_status()
            return response.json()["embedding"]

    async def store(self, entry: MemoryEntry) -> None:
        vector = entry.embedding if entry.embedding else await self.embed(entry.text)

        row = {
            "id": entry.id,
            "text": entry.text,
            "outcome": entry.outcome.value,
            "score": float(entry.score),
            "remark": entry.remark or "",
            "iteration": entry.iteration,
            "ts": entry.ts.isoformat(),
            "vector": vector,
        }

        existing = await asyncio.to_thread(
            lambda: self._table.search().where(f"id = '{entry.id}'").limit(1).to_list()
        )
        if existing:
            await asyncio.to_thread(
                lambda: self._table.delete(f"id = '{entry.id}'")
            )

        await asyncio.to_thread(self._table.add, [row])

    async def retrieve(
        self, query: str, k: int = 5, include_failures: bool = True
    ) -> list[MemoryEntry]:
        count = await asyncio.to_thread(self._table.count_rows)
        if count == 0:
            return []

        vector = await self.embed(query)

        def _search() -> list[dict]:
            q = self._table.search(vector).limit(k)
            if not include_failures:
                q = q.where(f"outcome != '{OutcomeType.MISTAKE.value}'")
            return q.to_list()

        rows = await asyncio.to_thread(_search)
        return [self._row_to_entry(r) for r in rows]

    async def is_duplicate_failure(self, text: str, threshold: float = 0.92) -> bool:
        count = await asyncio.to_thread(self._table.count_rows)
        if count == 0:
            return False

        vector = await self.embed(text)

        def _search() -> list[dict]:
            return (
                self._table.search(vector)
                .where(f"outcome = '{OutcomeType.MISTAKE.value}'")
                .limit(1)
                .to_list()
            )

        rows = await asyncio.to_thread(_search)
        if not rows:
            return False

        distance = rows[0].get("_distance", 1.0)
        cosine_sim = 1.0 - distance
        return cosine_sim > threshold

    async def get_best_wins(self, k: int = 3) -> list[MemoryEntry]:
        """Return the k highest-scoring wins, ordered by score descending."""
        count = await asyncio.to_thread(self._table.count_rows)
        if count == 0:
            return []

        def _fetch() -> list[dict]:
            import pyarrow.compute as pc
            tbl = self._table.to_arrow()
            mask = pc.equal(tbl.column("outcome"), OutcomeType.WIN.value)
            wins = tbl.filter(mask)
            if wins.num_rows == 0:
                return []
            indices = pc.sort_indices(wins, sort_keys=[("score", "descending")])
            top = wins.take(indices[:k])
            return top.to_pylist()

        rows = await asyncio.to_thread(_fetch)
        return [self._row_to_entry(r) for r in rows]

    async def is_building_on_win(self, text: str, similarity_floor: float = 0.80) -> bool:
        """True if text is semantically close to a past win — it may be an incremental improvement."""
        count = await asyncio.to_thread(self._table.count_rows)
        if count == 0:
            return False

        vector = await self.embed(text)

        def _search() -> list[dict]:
            return (
                self._table.search(vector)
                .where(f"outcome = '{OutcomeType.WIN.value}'")
                .limit(1)
                .to_list()
            )

        rows = await asyncio.to_thread(_search)
        if not rows:
            return False

        distance = rows[0].get("_distance", 1.0)
        cosine_sim = 1.0 - distance
        return cosine_sim > similarity_floor

    async def top_failures(self, context: str, k: int = 5) -> list[MemoryEntry]:
        count = await asyncio.to_thread(self._table.count_rows)
        if count == 0:
            return []

        vector = await self.embed(context)

        def _search() -> list[dict]:
            return (
                self._table.search(vector)
                .where(f"outcome = '{OutcomeType.MISTAKE.value}'")
                .limit(k)
                .to_list()
            )

        rows = await asyncio.to_thread(_search)
        return [self._row_to_entry(r) for r in rows]

    def _row_to_entry(self, row: dict) -> MemoryEntry:
        import datetime

        return MemoryEntry(
            id=row["id"],
            text=row["text"],
            outcome=OutcomeType(row["outcome"]),
            score=float(row["score"]),
            remark=row["remark"] or None,
            embedding=None,
            ts=datetime.datetime.fromisoformat(row["ts"]),
            iteration=int(row["iteration"]),
        )
