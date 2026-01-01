import os
import re
import sqlite3
from typing import Dict, Optional

from fastapi import FastAPI
from pydantic import BaseModel
from openai import OpenAI

app = FastAPI(title="Synq Backend")
client = OpenAI()

MODEL = "gpt-4o-mini"

# --- Simple SQLite memory store ---
DB_PATH = os.getenv("SYNQ_DB_PATH", "synq.db")
conn = sqlite3.connect(DB_PATH, check_same_thread=False)
conn.execute(
    """
    CREATE TABLE IF NOT EXISTS memory (
        session_id TEXT NOT NULL,
        key TEXT NOT NULL,
        value TEXT NOT NULL,
        updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        PRIMARY KEY (session_id, key)
    )
    """
)
conn.commit()

def set_memory(session_id: str, key: str, value: str) -> None:
    conn.execute(
        """
        INSERT INTO memory(session_id, key, value)
        VALUES(?, ?, ?)
        ON CONFLICT(session_id, key) DO UPDATE SET value=excluded.value, updated_at=CURRENT_TIMESTAMP
        """,
        (session_id, key, value),
    )
    conn.commit()

def get_memory(session_id: str) -> Dict[str, str]:
    cur = conn.execute(
        "SELECT key, value FROM memory WHERE session_id = ?",
        (session_id,),
    )
    return {k: v for (k, v) in cur.fetchall()}

# --- Request/Response models ---
class ChatRequest(BaseModel):
    session_id: str
    message: str

# --- Lightweight “remember” parser (keeps it simple) ---
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
    # 1) Save memory if user is telling you to remember
    updates = extract_memory_updates(req.message)
    saved_lines = []
    for k, v in updates.items():
        set_memory(req.session_id, k, v)
        saved_lines.append(f"{k} = {v}")

    # 2) Load memory and inject into system context
    mem = get_memory(req.session_id)
    mem_text = "\n".join([f"- {k}: {v}" for k, v in mem.items()]) if mem else "(none)"

    system = (
        "You are Synq, an action-oriented personal AI agent.\n"
        "You remember user preferences and use them to help execute tasks.\n"
        "Before any booking/payment, confirm details.\n\n"
        f"User memory:\n{mem_text}\n"
    )

    # 3) If user just saved memory, acknowledge quickly + continue
    # (We still let the model respond to keep it natural.)
    user_msg = req.message
    if saved_lines:
        user_msg = (
            req.message
            + "\n\n(You have successfully saved these memory fields: "
            + "; ".join(saved_lines)
            + ")"
        )

    resp = client.responses.create(
        model=MODEL,
        input=[
            {"role": "system", "content": system},
            {"role": "user", "content": user_msg},
        ],
    )

    return {"reply": (resp.output_text or "").strip(), "memory_saved": saved_lines}



