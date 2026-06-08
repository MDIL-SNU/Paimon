import sqlite3
from datetime import datetime
from pathlib import Path
from typing import TypedDict, cast

from paimon_web import cfg
from paimon_web.observability.models import RunStatusLiteral
from paimon_web.util.log import debug, info


class RunRow(TypedDict):
    env_id: str
    status: str
    task_name: str
    task: str
    agent: str
    total_cost: float
    created_at: str
    updated_at: str
    working_dir: str


class RunIndex:
    """SQLite-based index for fast run lookups"""

    def __init__(self, db_path: str):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        debug(f"[index] Initializing RunIndex at {self.db_path}")
        self._init_db()
        info(f"[index] RunIndex initialized at {self.db_path}")

    def _init_db(self):
        """Initialize database schema"""
        debug("[index] Initializing database schema")
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS runs (
                env_id TEXT PRIMARY KEY,
                task_name TEXT,
                task TEXT,
                status TEXT,
                agent TEXT,
                total_cost REAL,
                created_at TEXT,
                updated_at TEXT,
                working_dir TEXT
            )
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_updated_at ON runs(updated_at DESC)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_status ON runs(status)
        """)
        conn.commit()
        conn.close()
        debug("[index] Database schema initialized")

    def upsert_run(
        self,
        env_id: str,
        task_name: str,
        task: str,
        status: RunStatusLiteral,
        agent: str,
        total_cost: float,
        working_dir: str,
    ):
        """Insert or update run information"""
        debug(f"[index] Upserting run {env_id} (status={status})")
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()

        now = datetime.now().isoformat(timespec='seconds')

        cursor.execute(
            "SELECT created_at FROM runs WHERE env_id = ?",
            (env_id,)
        )
        result = cursor.fetchone()
        created_at = result[0] if result else now

        cursor.execute("""
            INSERT OR REPLACE INTO runs
            (env_id, task_name, task, status, agent, total_cost, created_at, updated_at, working_dir)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (env_id, task_name, task, status, agent, total_cost, created_at, now, working_dir))

        conn.commit()
        conn.close()
        debug(f"[index] Upserted run {env_id}")

    def list_runs(self, limit: int | None = None) -> list[RunRow]:
        """List all runs from index, ordered by updated_at"""
        debug(f"[index] Listing runs (limit={limit})")
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        query = "SELECT * FROM runs ORDER BY updated_at DESC"
        if limit:
            query += f" LIMIT {limit}"

        cursor.execute(query)
        rows = cursor.fetchall()
        conn.close()

        debug(f"[index] Retrieved {len(rows)} runs from index")
        return [cast(RunRow, dict(row)) for row in rows]

    def get_run(self, env_id: str) -> RunRow | None:
        """Get single run by env_id"""
        debug(f"[index] Getting run {env_id}")
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute("SELECT * FROM runs WHERE env_id = ?", (env_id,))
        row = cursor.fetchone()
        conn.close()

        if row:
            debug(f"[index] Found run {env_id}")
        else:
            debug(f"[index] Run {env_id} not found")
        return cast(RunRow, dict(row)) if row else None

    def delete_run(self, env_id: str):
        """Remove run from index"""
        debug(f"[index] Deleting run {env_id}")
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        cursor.execute("DELETE FROM runs WHERE env_id = ?", (env_id,))
        conn.commit()
        conn.close()
        debug(f"[index] Deleted run {env_id}")

    def list_runs_by_status(
        self,
        status_list: list[RunStatusLiteral],
    ) -> list[RunRow]:
        if not status_list:
            return []

        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()

        placeholders = ",".join("?" for _ in status_list)
        query = f"""
            SELECT * FROM runs
            WHERE status IN ({placeholders})
        """

        cur.execute(query, status_list)
        rows = cur.fetchall()
        conn.close()

        return [cast(RunRow, dict(row)) for row in rows]

    def list_runs_filtered(
        self,
        status: str | None = None,
        search: str | None = None,
        date_from: str | None = None,
        date_to: str | None = None,
        limit: int | None = None,
    ) -> list[RunRow]:
        """List runs with optional filters."""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()

        conditions = []
        params: list = []

        if status:
            conditions.append("status = ?")
            params.append(status)

        if search:
            conditions.append("(task_name LIKE ? OR task LIKE ?)")
            params.extend([f"%{search}%", f"%{search}%"])

        if date_from:
            conditions.append("updated_at >= ?")
            params.append(date_from)

        if date_to:
            conditions.append("updated_at <= ?")
            params.append(date_to + "T23:59:59")

        query = "SELECT * FROM runs"
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        query += " ORDER BY updated_at DESC"

        if limit:
            query += f" LIMIT {limit}"

        cur.execute(query, params)
        rows = cur.fetchall()
        conn.close()

        return [cast(RunRow, dict(row)) for row in rows]

    def get_distinct_statuses(self) -> list[str]:
        """Get all distinct status values in DB."""
        conn = sqlite3.connect(str(self.db_path))
        cur = conn.cursor()
        cur.execute("SELECT DISTINCT status FROM runs ORDER BY status")
        rows = cur.fetchall()
        conn.close()
        return [row[0] for row in rows]


info("[index] Initializing DB")
run_index: RunIndex = RunIndex(cfg.index_db_path)
info("[index] DB initialized successfully")
