import os
import re
from typing import Dict

from fastapi import FastAPI
from pydantic import BaseModel
from openai import OpenAI

import psycopg

app = FastAPI(title="Synq Backend")
client = OpenAI()

MODEL = "gpt-4o-mini"

DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL is not set in this service. Add Railway Postgres and reference DATABASE_URL.")

def get_conn():
    return psycopg.connect(DATABASE_URL)

def init_db():
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS memory (
                    session_id TEXT NOT NULL,
                    key TEXT NOT NULL,
                    value TEXT NOT NULL,
                    updated_at TIMESTAMPTZ DEFAULT NOW(),
                    PRIMARY KEY (session_id, key)
                );
                """
            )
        conn.commit()

init_db()

def set_memory(session_id: str, key: str, value: str) -> None:
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO memory(session_id, key, value)
                VALUES (%s, %s, %s)
                ON CONFLICT (session_id, key)
                DO UPDATE SET value = EXCLUDED.value, updated_at = NOW();
                """,
                (session_id, key, value),
            )
        conn.commit()

def get_memory(session_id: str) -> Dict[str, str]:
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT key, value FROM memory WHERE session_id = %s;",
                (session_id,),
            )
            rows = cur.fetchall()
    return {k: v for (k, v) in rows}

class ChatRequest(BaseModel):
    session_id: str
    message: str

def extract_memory_updates(text: str) -> Dict[str, str]:
    t = text.strip()
    patterns = [
        (r"^remember\s+my\s+office\s+is\s+(.+)$", "office_address"),
        (r"^remember\s+my\s+home\s+is\s+(.+)$", "home_address"),
        (r"^(?:remember\s+)?my\s+favorite\s+pizza\s+is\s+(.+)$", "fav_pizza_order"),
        (r"^(?:remember\s+)?my\s+favorite\s+pizza\s+restaurant\s+is\s+(.+)$", "fav_pizza_restaurant"),
    ]
    updates: Dict[str, str] = {}
    for pat, key in patterns:
        m = re.match(pat, t, flags=re.IGNORECASE)
        if m:
            updates[key] = m.group(1).strip()
    return updates

@app.get("/health")
def health():
    return {"ok": True, "service": "synq"}

@app.get("/memory/{session_id}")
def read_memory(session_id: str):
    return {"session_id": session_id, "memory": get_memory(session_id)}

@app.post("/chat")
def chat(req: ChatRequest):
    # Save memory updates
    updates = extract_memory_updates(req.message)
    saved_lines = []
    for k, v in updates.items():
        set_memory(req.session_id, k, v)
        saved_lines.append(f"{k} = {v}")

    # Load memory
    mem = get_memory(req.session_id)
    mem_text = "\n".join([f"- {k}: {v}" for k, v in mem.items()]) if mem else "(none)"

    system = (
        "You are Synq, an action-oriented personal AI agent.\n"
        "You remember user preferences and use them to help execute tasks.\n"
        "Before any booking/payment, confirm details.\n\n"
        f"User memory:\n{mem_text}\n"
    )

    resp = client.responses.create(
        model=MODEL,
        input=[
            {"role": "system", "content": system},
            {"role": "user", "content": req.message},
        ],
    )

    return {"reply": (resp.output_text or "").strip(), "memory_saved": saved_lines}





