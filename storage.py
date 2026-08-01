"""Prediction history storage.

Uses Postgres when a DATABASE_URL env var is set (persists across
redeploys/restarts — needed on free hosting tiers with ephemeral disks),
otherwise falls back to a local JSON file. Also falls back to the JSON
file if the DB connection fails for any reason (bad URL, network issue),
so a database misconfiguration degrades the history feature instead of
crashing the whole site.
"""
import json
import os
import sys
from datetime import datetime, timezone
from urllib.parse import urlsplit, urlunsplit

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
HISTORY_PATH = os.path.join(_BASE_DIR, "prediction_history.json")
MAX_HISTORY = 500

_RAW_DATABASE_URL = os.environ.get("DATABASE_URL")


def _clean_dsn(url: str) -> str:
    """Strips query params libpq doesn't understand (e.g. Supabase's
    "?pgbouncer=true", which is a hint for some ORMs, not a real
    connection option — libpq rejects unknown params outright)."""
    parts = urlsplit(url)
    return urlunsplit((parts.scheme, parts.netloc, parts.path, "", ""))


_db_ready = False

if _RAW_DATABASE_URL:
    try:
        import psycopg2
        import psycopg2.extras

        DATABASE_URL = _clean_dsn(_RAW_DATABASE_URL)

        def _get_conn():
            return psycopg2.connect(DATABASE_URL, sslmode="require")

        with _get_conn() as _conn:
            with _conn.cursor() as _cur:
                _cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS predictions (
                        id SERIAL PRIMARY KEY,
                        text TEXT NOT NULL,
                        is_fake BOOLEAN NOT NULL,
                        confidence DOUBLE PRECISION,
                        ml_probability DOUBLE PRECISION,
                        matched_patterns JSONB,
                        created_at TIMESTAMPTZ NOT NULL
                    )
                    """
                )
            _conn.commit()
        _db_ready = True
    except Exception as exc:
        print(
            f"[storage] DATABASE_URL set but connection/init failed, "
            f"falling back to local JSON file: {exc!r}",
            file=sys.stderr,
        )
        _db_ready = False

if _db_ready:

    def load_history():
        with _get_conn() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(
                    "SELECT text, is_fake, confidence, ml_probability, matched_patterns, created_at "
                    "FROM predictions ORDER BY id ASC"
                )
                rows = cur.fetchall()
        return [
            {
                "text": r["text"],
                "is_fake": r["is_fake"],
                "confidence": r["confidence"],
                "ml_probability": r["ml_probability"],
                "matched_patterns": r["matched_patterns"] or [],
                "created_at": r["created_at"].isoformat(),
            }
            for r in rows
        ]

    def append_history(item):
        with _get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO predictions "
                    "(text, is_fake, confidence, ml_probability, matched_patterns, created_at) "
                    "VALUES (%s, %s, %s, %s, %s, %s)",
                    (
                        item["text"],
                        item["is_fake"],
                        item["confidence"],
                        item["ml_probability"],
                        json.dumps(item["matched_patterns"]),
                        item["created_at"],
                    ),
                )
            conn.commit()

    def clear_history():
        with _get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM predictions")
            conn.commit()

else:

    def _read_all():
        try:
            with open(HISTORY_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data if isinstance(data, list) else []
        except Exception:
            return []

    def _write_all(history):
        try:
            with open(HISTORY_PATH, "w", encoding="utf-8") as f:
                json.dump(history[-MAX_HISTORY:], f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    def load_history():
        return _read_all()

    def append_history(item):
        history = _read_all()
        history.append(item)
        _write_all(history)

    def clear_history():
        _write_all([])


def new_history_item(text, result):
    return {
        "text": text,
        "is_fake": result["is_fake"],
        "confidence": result["confidence"],
        "ml_probability": result["ml_probability"],
        "matched_patterns": result["matched_patterns"],
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
