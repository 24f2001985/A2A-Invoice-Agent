
import sqlite3
import json
from pathlib import Path
from datetime import datetime, timezone

DB_PATH = Path(__file__).parent / "invoice_agent.db"


def get_connection():
    conn = sqlite3.connect(DB_PATH, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    return conn


def initialize_db():
    with get_connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS tasks (
                task_id TEXT PRIMARY KEY,
                principal TEXT NOT NULL,
                state TEXT NOT NULL,
                data TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
        """)

        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_tasks_principal
            ON tasks(principal)
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS idempotency (
                principal TEXT NOT NULL,
                message_id TEXT NOT NULL,
                body_hash TEXT NOT NULL,
                task_id TEXT NOT NULL,
                PRIMARY KEY (principal, message_id)
            )
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS decision_cache (
                package_hash TEXT PRIMARY KEY,
                decision_json TEXT NOT NULL
            )
        """)


def save_task(task_id, principal, state, data):
    now = datetime.now(timezone.utc).isoformat()

    with get_connection() as conn:
        existing = conn.execute("""
            SELECT principal FROM tasks WHERE task_id = ?
        """, (task_id,)).fetchone()

        if existing and existing["principal"] != principal:
            raise ValueError("Task ownership conflict")

        conn.execute("""
            INSERT INTO tasks (
                task_id, principal, state, data, created_at, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(task_id) DO UPDATE SET
                state = excluded.state,
                data = excluded.data,
                updated_at = excluded.updated_at
        """, (
            task_id,
            principal,
            state,
            json.dumps(data, separators=(",", ":"), ensure_ascii=False),
            now,
            now
        ))


def get_task(task_id, principal):
    with get_connection() as conn:
        row = conn.execute("""
            SELECT data FROM tasks
            WHERE task_id = ? AND principal = ?
        """, (task_id, principal)).fetchone()

    return json.loads(row["data"]) if row else None


def get_task_record(task_id, principal):
    with get_connection() as conn:
        row = conn.execute("""
            SELECT task_id, principal, state, data
            FROM tasks
            WHERE task_id = ? AND principal = ?
        """, (task_id, principal)).fetchone()

    if not row:
        return None

    record = dict(row)
    record["data"] = json.loads(record["data"])
    return record


def list_tasks(principal):
    with get_connection() as conn:
        rows = conn.execute("""
            SELECT data FROM tasks
            WHERE principal = ?
            ORDER BY created_at DESC
        """, (principal,)).fetchall()

    return [json.loads(row["data"]) for row in rows]


def find_idempotency(principal, message_id):
    with get_connection() as conn:
        row = conn.execute("""
            SELECT body_hash, task_id
            FROM idempotency
            WHERE principal = ? AND message_id = ?
        """, (principal, message_id)).fetchone()

    return dict(row) if row else None


def save_idempotency(principal, message_id, body_hash, task_id):
    with get_connection() as conn:
        conn.execute("""
            INSERT INTO idempotency (
                principal, message_id, body_hash, task_id
            )
            VALUES (?, ?, ?, ?)
        """, (principal, message_id, body_hash, task_id))


def get_cached_decision(package_hash):
    with get_connection() as conn:
        row = conn.execute("""
            SELECT decision_json FROM decision_cache
            WHERE package_hash = ?
        """, (package_hash,)).fetchone()

    return json.loads(row["decision_json"]) if row else None


def cache_decision(package_hash, decision):
    with get_connection() as conn:
        conn.execute("""
            INSERT INTO decision_cache (package_hash, decision_json)
            VALUES (?, ?)
            ON CONFLICT(package_hash) DO NOTHING
        """, (
            package_hash,
            json.dumps(decision, separators=(",", ":"), ensure_ascii=False)
        ))


def cancel_task(task_id, principal):
    conn = get_connection()

    try:
        conn.execute("BEGIN IMMEDIATE")

        row = conn.execute(
            "SELECT state, data FROM tasks WHERE task_id = ? AND principal = ?",
            (task_id, principal)
        ).fetchone()

        if row is None:
            conn.rollback()
            return None

        terminal_states = {
            "TASK_STATE_COMPLETED",
            "TASK_STATE_CANCELED",
            "TASK_STATE_REJECTED"
        }

        if row["state"] in terminal_states:
            conn.rollback()
            raise ValueError("Task is already in a terminal state")

        task = json.loads(row["data"])

        now = datetime.now(timezone.utc).isoformat()

        task["status"] = {
            "state": "TASK_STATE_CANCELED",
            "timestamp": now
        }

        conn.execute(
            """
            UPDATE tasks
            SET state = ?, data = ?, updated_at = ?
            WHERE task_id = ? AND principal = ?
            """,
            (
                "TASK_STATE_CANCELED",
                json.dumps(task),
                now,
                task_id,
                principal
            )
        )

        conn.commit()
        return task

    except Exception:
        conn.rollback()
        raise

    finally:
        conn.close()



def reserve_idempotency(principal, message_id, body_hash, task_id):
    from protocol import create_task

    with get_connection() as conn:
        conn.execute("BEGIN IMMEDIATE")

        existing = conn.execute("""
            SELECT body_hash, task_id
            FROM idempotency
            WHERE principal = ? AND message_id = ?
        """, (principal, message_id)).fetchone()

        if existing:
            if existing["body_hash"] != body_hash:
                raise ValueError("IDEMPOTENCY_CONFLICT")

            return {
                "created": False,
                "body_hash": existing["body_hash"],
                "task_id": existing["task_id"]
            }

        task = create_task(task_id=task_id)
        now = datetime.now(timezone.utc).isoformat()

        conn.execute("""
            INSERT INTO tasks (
                task_id, principal, state, data, created_at, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            task_id,
            principal,
            task["status"]["state"],
            json.dumps(task, separators=(",", ":"), ensure_ascii=False),
            now,
            now
        ))

        conn.execute("""
            INSERT INTO idempotency (
                principal, message_id, body_hash, task_id
            )
            VALUES (?, ?, ?, ?)
        """, (principal, message_id, body_hash, task_id))

        return {
            "created": True,
            "body_hash": body_hash,
            "task_id": task_id
        }
