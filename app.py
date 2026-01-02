import os
import re
from typing import Dict, Optional, Any

from fastapi import FastAPI
from pydantic import BaseModel
from openai import OpenAI

import psycopg

app = FastAPI(title="Synq Backend")
client = OpenAI()

MODEL = "gpt-4o-mini"

DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL is not set in this service. Ensure Railway Postgres DATABASE_URL is referenced.")

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
            cur.execute("SELECT key, value FROM memory WHERE session_id = %s;", (session_id,))
            rows = cur.fetchall()
    return {k: v for (k, v) in rows}

class ChatRequest(BaseModel):
    session_id: str
    message: str

# -------------------------
# 1) Memory extraction (simple rules)
# -------------------------
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

# -------------------------
# 2) Action planning (rule-based, reliable)
# -------------------------
def plan_action(user_text: str, mem: Dict[str, str]) -> Optional[Dict[str, Any]]:
    t = user_text.strip().lower()

    # BOOK RIDE intents
    if any(phrase in t for phrase in ["book a cab", "book cab", "book a car", "book grab", "grab me", "get me a cab", "book ride", "book a ride"]):
        # Destination resolution
        to = None
        if "to my office" in t or "to office" in t:
            to = mem.get("office_address")
        elif "to my home" in t or "to home" in t:
            to = mem.get("home_address")
        else:
            # Try “to <place>”
            m = re.search(r"\bto\s+(.+)$", user_text, flags=re.IGNORECASE)
            if m:
                to = m.group(1).strip()

        return {
            "action": "BOOK_RIDE",
            "args": {
                "provider": "GRAB",
                "from": "current_location",
                "to": to,
            },
            "needs_confirmation": True,
            "missing": ["to"] if not to else [],
        }

    # ORDER PIZZA intents
    if any(phrase in t for phrase in ["order pizza", "book pizza", "get me pizza", "pizza from", "order my favorite pizza", "order favourite pizza"]):
        restaurant = mem.get("fav_pizza_restaurant")
        order = mem.get("fav_pizza_order")

        # If user specifies restaurant “from <x>”
        m = re.search(r"\bfrom\s+(.+)$", user_text, flags=re.IGNORECASE)
        if m:
            restaurant = m.group(1).strip()

        return {
            "action": "ORDER_PIZZA",
            "args": {
                "provider": "FOODPANDA",
                "restaurant": restaurant,
                "order": order,
                "deliver_to": mem.get("home_address"),
            },
            "needs_confirmation": True,
            "missing": [k for k in ["restaurant", "order"] if (restaurant if k=="restaurant" else order) is None],
        }

    return None

@app.get("/health")
def health():
    return {"ok": True, "service": "synq"}

@app.get("/memory/{session_id}")
def read_memory(session_id: str):
    return {"session_id": session_id, "memory": get_memory(session_id)}

@app.post("/chat")
def chat(req: ChatRequest):
    # Save memory if applicable
    updates = extract_memory_updates(req.message)
    memory_saved = []
    for k, v in updates.items():
        set_memory(req.session_id, k, v)
        memory_saved.append(f"{k} = {v}")

    # Load memory
    mem = get_memory(req.session_id)

    # Create an action plan if the user is asking to DO something
    plan = plan_action(req.message, mem)

    # System prompt includes memory and “action style”
    mem_text = "\n".join([f"- {k}: {v}" for k, v in mem.items()]) if mem else "(none)"
    system = (
        "You are Synq, an action-oriented personal AI agent.\n"
        "You remember user preferences and use them to help execute tasks.\n"
        "If a plan is present, speak briefly and ask for confirmation.\n"
        "Never claim you actually booked/paid. You only prepare actions.\n\n"
        f"User memory:\n{mem_text}\n"
    )

    # If plan exists, guide the user to confirm + request missing info
    if plan:
        missing = plan.get("missing", [])
        if missing:
            # Ask only for what’s missing
            if plan["action"] == "BOOK_RIDE" and "to" in missing:
                user_msg = "User wants to book a ride but destination is missing. Ask where they want to go (office/home or specify a place)."
            elif plan["action"] == "ORDER_PIZZA":
                user_msg = "User wants to order pizza but favorites are missing. Ask for restaurant and pizza order (size/toppings)."
            else:
                user_msg = "Ask for the missing details needed to proceed."
        else:
            # Confirm details
            if plan["action"] == "BOOK_RIDE":
                user_msg = f"Confirm booking a Grab ride to: {plan['args'].get('to')}. Ask for confirmation."
            else:
                user_msg = (
                    f"Confirm ordering pizza via Foodpanda from: {plan['args'].get('restaurant')}, "
                    f"order: {plan['args'].get('order')}. Ask for confirmation."
                )
    else:
        user_msg = req.message

    resp = client.responses.create(
        model=MODEL,
        input=[
            {"role": "system", "content": system},
            {"role": "user", "content": user_msg},
        ],
    )

    return {
        "reply": (resp.output_text or "").strip(),
        "memory_saved": memory_saved,
        "plan": plan,  # <-- THIS is the action plan JSON (or null)
    }
