from __future__ import annotations

import json
import sqlite3
import threading
from typing import Optional

from shared.types import AgentState


class StateMemory:
    def __init__(self, db_path: str) -> None:
        self._db_path = db_path
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self.init_db()

    def init_db(self) -> None:
        with self._lock:
            self._conn.execute(
                "CREATE TABLE IF NOT EXISTS state (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
            )
            self._conn.commit()

    def save_state(self, state: AgentState) -> None:
        payload = json.dumps(
            {
                "iteration": state.iteration,
                "baseline_score": state.baseline_score,
                "working_commit": state.working_commit,
                "run_id": state.run_id,
            }
        )
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO state (key, value) VALUES ('agent_state', ?)",
                (payload,),
            )
            self._conn.commit()

    def load_state(self) -> Optional[AgentState]:
        with self._lock:
            row = self._conn.execute(
                "SELECT value FROM state WHERE key = 'agent_state'"
            ).fetchone()

        if row is None:
            return None

        data = json.loads(row[0])
        return AgentState(
            iteration=data["iteration"],
            baseline_score=data["baseline_score"],
            working_commit=data["working_commit"],
            run_id=data["run_id"],
        )
