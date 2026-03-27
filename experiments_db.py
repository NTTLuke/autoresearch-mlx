"""
SQLite-backed experiment tracker for autoresearch.

Stores one row per experiment with the LM Studio model ID that drove the
decision, so results can be compared across different local models.
"""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path

DEFAULT_DB = Path(__file__).resolve().parent / "autoresearch.db"


def _connect(db_path: Path | str = DEFAULT_DB) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: Path | str = DEFAULT_DB) -> None:
    with _connect(db_path) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS experiments (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp    REAL    NOT NULL,
                model_id     TEXT    NOT NULL,
                val_bpb      REAL    NOT NULL,
                memory_gb    REAL    NOT NULL,
                status       TEXT    NOT NULL,
                description  TEXT    NOT NULL,
                config_json  TEXT    NOT NULL DEFAULT '{}',
                weights_path TEXT    NOT NULL DEFAULT ''
            )
        """)
        cols = [r[1] for r in conn.execute("PRAGMA table_info(experiments)").fetchall()]
        if "config_json" not in cols:
            conn.execute("ALTER TABLE experiments ADD COLUMN config_json TEXT NOT NULL DEFAULT '{}'")
        if "weights_path" not in cols:
            conn.execute("ALTER TABLE experiments ADD COLUMN weights_path TEXT NOT NULL DEFAULT ''")
        conn.commit()


def log_experiment(
    model_id: str,
    val_bpb: float,
    memory_gb: float,
    status: str,
    description: str,
    db_path: Path | str = DEFAULT_DB,
    config_json: str = "{}",
) -> int:
    init_db(db_path)
    with _connect(db_path) as conn:
        cur = conn.execute(
            "INSERT INTO experiments (timestamp, model_id, val_bpb, memory_gb, status, description, config_json) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (time.time(), model_id, val_bpb, memory_gb, status, description, config_json),
        )
        conn.commit()
        return cur.lastrowid  # type: ignore[return-value]


def get_experiments(
    model_id: str | None = None,
    db_path: Path | str = DEFAULT_DB,
) -> list[dict]:
    init_db(db_path)
    with _connect(db_path) as conn:
        if model_id:
            rows = conn.execute(
                "SELECT * FROM experiments WHERE model_id = ? ORDER BY timestamp",
                (model_id,),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM experiments ORDER BY timestamp"
            ).fetchall()
    return [dict(r) for r in rows]


def update_weights_path(row_id: int, weights_path: str, db_path: Path | str = DEFAULT_DB) -> None:
    with _connect(db_path) as conn:
        conn.execute(
            "UPDATE experiments SET weights_path = ? WHERE id = ?",
            (weights_path, row_id),
        )
        conn.commit()


def get_experiments_with_weights(db_path: Path | str = DEFAULT_DB) -> list[dict]:
    init_db(db_path)
    with _connect(db_path) as conn:
        rows = conn.execute(
            "SELECT * FROM experiments WHERE weights_path != '' ORDER BY val_bpb"
        ).fetchall()
    return [dict(r) for r in rows]


def get_experiment_by_id(experiment_id: int, db_path: Path | str = DEFAULT_DB) -> dict | None:
    init_db(db_path)
    with _connect(db_path) as conn:
        row = conn.execute(
            "SELECT * FROM experiments WHERE id = ?", (experiment_id,)
        ).fetchone()
    return dict(row) if row else None


def get_best_by_model(db_path: Path | str = DEFAULT_DB) -> list[dict]:
    init_db(db_path)
    with _connect(db_path) as conn:
        rows = conn.execute("""
            SELECT model_id,
                   MIN(val_bpb) AS best_val_bpb,
                   COUNT(*)     AS total_runs,
                   SUM(CASE WHEN status = 'keep' THEN 1 ELSE 0 END) AS kept,
                   SUM(CASE WHEN status = 'crash' THEN 1 ELSE 0 END) AS crashed
            FROM experiments
            GROUP BY model_id
            ORDER BY best_val_bpb
        """).fetchall()
    return [dict(r) for r in rows]
