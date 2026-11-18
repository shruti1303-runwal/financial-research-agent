"""
Saved answers, stored in Postgres next to the vectors.

A repeated question (same words, ignoring case/spacing/punctuation) is answered from here
instantly and uses no LLM quota. Answers built from live market data expire after a few
hours; answers built only from the reports are kept for a month.
"""
import json
import re

from sqlalchemy import text

from app.vectorstore import get_engine

LIVE_TOOLS = {"get_stock_quote", "get_price_history", "get_key_metrics"}
TTL_LIVE_HOURS = 6
TTL_REPORT_HOURS = 24 * 30

_ready = False


def _key(question: str) -> str:
    q = re.sub(r"\s+", " ", question.lower()).strip()
    return q.rstrip(" ?.!")


def _ensure_table():
    global _ready
    if _ready:
        return
    with get_engine().begin() as con:
        con.execute(text("""
            CREATE TABLE IF NOT EXISTS answer_cache (
                qkey       text PRIMARY KEY,
                payload    jsonb NOT NULL,
                expires_at timestamptz NOT NULL,
                created_at timestamptz NOT NULL DEFAULT now())"""))
    _ready = True


def get(question: str) -> dict | None:
    _ensure_table()
    with get_engine().connect() as con:
        row = con.execute(text(
            "SELECT payload, created_at FROM answer_cache WHERE qkey = :k AND expires_at > now()"),
            {"k": _key(question)}).first()
    if not row:
        return None
    payload = dict(row[0])
    payload.update(question=question, cached=True, cached_at=row[1].isoformat())
    return payload


def put(question: str, out: dict):
    _ensure_table()
    live = any(s["tool"] in LIVE_TOOLS for s in out.get("tool_calls", []))
    hours = TTL_LIVE_HOURS if live else TTL_REPORT_HOURS
    payload = {k: v for k, v in out.items() if k not in ("contexts", "question")}
    with get_engine().begin() as con:
        con.execute(text("""
            INSERT INTO answer_cache (qkey, payload, expires_at)
            VALUES (:k, CAST(:p AS jsonb), now() + make_interval(hours => :h))
            ON CONFLICT (qkey) DO UPDATE
              SET payload = EXCLUDED.payload, expires_at = EXCLUDED.expires_at, created_at = now()"""),
            {"k": _key(question), "p": json.dumps(payload, ensure_ascii=False), "h": hours})
