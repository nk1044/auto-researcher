from __future__ import annotations

import json
import sqlite3
import threading
from typing import Optional

from shared.types import IterationRecord, OutcomeType


class EpisodicMemory:
    def __init__(self, db_path: str) -> None:
        self._db_path = db_path
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self.init_db()

    def init_db(self) -> None:
        with self._lock:
            self._conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS iterations (
                    id                  TEXT PRIMARY KEY,
                    hypothesis          TEXT NOT NULL,
                    integrated_diff_hash TEXT NOT NULL,
                    subagent_contribs   TEXT NOT NULL,
                    score               REAL NOT NULL,
                    remark              TEXT,
                    outcome             TEXT NOT NULL,
                    baseline_before     REAL NOT NULL,
                    ts                  TEXT NOT NULL,
                    iteration           INTEGER NOT NULL
                );
                CREATE TABLE IF NOT EXISTS meta (
                    key   TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
                """
            )
            self._conn.commit()

    def record(self, record: IterationRecord) -> None:
        with self._lock:
            self._conn.execute(
                """
                INSERT OR REPLACE INTO iterations
                    (id, hypothesis, integrated_diff_hash, subagent_contribs,
                     score, remark, outcome, baseline_before, ts, iteration)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.id,
                    record.hypothesis,
                    record.integrated_diff_hash,
                    json.dumps(record.subagent_contribs),
                    record.score,
                    record.remark,
                    record.outcome.value,
                    record.baseline_before,
                    record.ts.isoformat(),
                    record.iteration,
                ),
            )
            self._conn.commit()

    def _row_to_record(self, row: sqlite3.Row) -> IterationRecord:
        import datetime

        return IterationRecord(
            id=row["id"],
            hypothesis=row["hypothesis"],
            integrated_diff_hash=row["integrated_diff_hash"],
            subagent_contribs=json.loads(row["subagent_contribs"]),
            score=row["score"],
            remark=row["remark"],
            outcome=OutcomeType(row["outcome"]),
            baseline_before=row["baseline_before"],
            ts=datetime.datetime.fromisoformat(row["ts"]),
            iteration=row["iteration"],
        )

    def get_recent(self, k: int = 20) -> list[IterationRecord]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM iterations ORDER BY iteration DESC LIMIT ?", (k,)
            ).fetchall()
        return [self._row_to_record(r) for r in rows]

    def get_by_iteration(self, n: int) -> Optional[IterationRecord]:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM iterations WHERE iteration = ?", (n,)
            ).fetchone()
        return self._row_to_record(row) if row else None

    def count(self) -> int:
        with self._lock:
            return self._conn.execute("SELECT COUNT(*) FROM iterations").fetchone()[0]

    def get_last(self) -> Optional[IterationRecord]:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM iterations ORDER BY iteration DESC LIMIT 1"
            ).fetchone()
        return self._row_to_record(row) if row else None
