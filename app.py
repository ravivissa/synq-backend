import os
import re
from typing import Dict, Optional, Any, List

from fastapi import FastAPI
from pydantic import BaseModel
from openai import OpenAI
import psycopg

# -------------------------
# App setup
# -------------------------
app = FastAPI(title="Synq Backend")
client = OpenAI()

MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise RuntimeError(
        "DATABASE_URL is not set. In Railway, add Postgres and reference DATABASE_URL into synq-backend variables."
    )

# -------------------------
# DB helpers (Postgres)
# -------------------------
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

def delete_memory(session_id: str, key: str) -> None:
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM memory WHERE session_id=%s AND key=%s;", (session_id, key))
        conn.commit()

def get_memory(session_id: str) -> Dict[str, str]:
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT key, value FROM memory WHERE session_id=%s;", (session_id,))
            rows = cur.fetchall()
    return {k: v for (k, v) in rows}

# -------------------------
# Models
# -------------------------
class ChatRequest(BaseModel):
    session_id: str
    message: str

# -------------------------
# Lists & helpers
# -------------------------
TOP_RIDE_PROVIDERS = ["UBER", "DIDI", "LYFT", "GRAB", "BOLT"]

def normalize_provider(name: Optional[str]) -> Optional[str]:
    if not name:
        return None
    n = name.strip().lower()
    mapping = {
        "uber": "UBER",
        "didi": "DIDI",
        "di di": "DIDI",
        "lyft": "LYFT",
        "grab": "GRAB",
        "bolt": "BOLT",
        "auto": "AUTO",
    }
    return mapping.get(n, name.strip().upper())

def choose_ride_provider(user_text: str, mem: Dict[str, str]) -> str:
    # If user explicitly mentions provider in the request, that wins
    lt = user_text.lower()
    for p in ["uber", "didi", "lyft", "grab", "bolt"]:
        if p in lt:
            return normalize_provider(p) or "UBER"

    # Else use saved preference
    pref = normalize_provider(mem.get("ride_provider"))
    if pref and pref in TOP_RIDE_PROVIDERS:
        return pref

    # Default
    return "UBER"

def parse_csv_list(s: Optional[str]) -> List[str]:
    if not s:
        return []
    return [x.strip().lower() for x in s.split(",") if x.strip()]

def add_avoid_items(existing_csv: Optional[str], new_items: List[str]) -> str:
    existing = set(parse_csv_list(existing_csv))
    for item in new_items:
        it = (item or "").strip().lower()
        if it:
            existing.add(it)
    return ", ".join(sorted(existing))

def contains_any(text: str, items: List[str]) -> List[str]:
    t = (text or "").lower()
    hits = []
    for it in items:
        if it and it.lower() in t:
            hits.append(it.lower())
    return hits

def infer_default_avoid_from_diet_and_religion(mem: Dict[str, str]) -> List[str]:
    diet = (mem.get("diet") or "").lower()
    religion = (mem.get("religion") or "").lower()

    defaults: List[str] = []

    # If vegetarian/vegan: avoid obvious meats
    if diet in ["vegetarian", "vegan"]:
        defaults += ["pepperoni", "pork", "bacon", "ham", "beef", "chicken", "sausage", "salami", "meat"]
    # If vegan: avoid dairy (simple)
    if diet == "vegan":
        defaults += ["cheese", "mozzarella", "milk", "butter", "cream", "yogurt"]

    # Conservative religion defaults (only if user explicitly stated religion)
    if religion == "hindu":
        defaults += ["beef"]
    if religion == "muslim":
        defaults += ["pork", "bacon", "ham"]
    if religion == "jain":
        defaults += ["onion", "garlic", "potato"]

    # unique preserve order
    seen = set()
    out = []
    for d in defaults:
        dl = d.lower()
        if dl not in seen:
            seen.add(dl)
            out.append(dl)
    return out

# -------------------------
# Memory parsing (updates + deletes)
# -------------------------
def extract_memory_updates_and_deletes(text: str, mem: Dict[str, str]) -> Dict[str, Any]:
    """
    Returns:
      {
        "updates": {key: value},
        "deletes": [key, ...],
      }
    """
    t = text.strip()

    updates: Dict[str, str] = {}
    deletes: List[str] = []

    # Basic remembers (locations + favorites)
    patterns = [
        (r"^remember\s+my\s+office\s+is\s+(.+)$", "office_address"),
        (r"^remember\s+my\s+home\s+is\s+(.+)$", "home_address"),
        (r"^remember\s+my\s+favorite\s+pizza\s+restaurant\s+is\s+(.+)$", "fav_pizza_restaurant"),
        (r"^(?:remember\s+)?my\s+favorite\s+pizza\s+is\s+(.+)$", "fav_pizza_order"),
    ]
    for pat, key in patterns:
        m = re.match(pat, t, flags=re.IGNORECASE)
        if m:
            updates[key] = m.group(1).strip()

    # Preferred ride provider
    m_r = re.match(r"^remember\s+my\s+ride\s+(app|provider)\s+is\s+(.+)$", t, flags=re.IGNORECASE)
    if m_r:
        prov = normalize_provider(m_r.group(2))
        if prov == "AUTO" or prov in TOP_RIDE_PROVIDERS:
            updates["ride_provider"] = "AUTO" if prov == "AUTO" else prov

    # Diet
    if re.search(r"\b(i am|i'm)\s+vegetarian\b", t, flags=re.IGNORECASE):
        updates["diet"] = "vegetarian"
    if re.search(r"\b(i am|i'm)\s+vegan\b", t, flags=re.IGNORECASE):
        updates["diet"] = "vegan"
    if re.search(r"\b(i eat\s+non-veg|i am\s+non-veg|i eat meat)\b", t, flags=re.IGNORECASE):
        updates["diet"] = "non_veg"

    # Religion (only if user explicitly says it)
    m_rel = re.search(r"\b(i am|i'm)\s+(hindu|muslim|christian|jain|sikh|buddhist)\b", t, flags=re.IGNORECASE)
    if m_rel:
        updates["religion"] = m_rel.group(2).strip().lower()

    # Avoid items: "I don't eat X" / "I do not eat X"
    # Keep it simple: capture after "eat"
    m_avoid = re.search(r"\b(i (?:do not|don't) eat)\s+(.+)$", t, flags=re.IGNORECASE)
    if m_avoid:
        item = m_avoid.group(2).strip().rstrip(".")
        if len(item) > 60:
            item = item[:60].strip()
        updates["avoid_items"] = add_avoid_items(mem.get("avoid_items"), [item])

    # Update commands (overwrite)
    m_upd = re.search(r"^(change|update)\s+my\s+favorite\s+pizza\s+(to|as)\s+(.+)$", t, flags=re.IGNORECASE)
    if m_upd:
        updates["fav_pizza_order"] = m_upd.group(3).strip()

    m_upd_r = re.search(r"^(change|update)\s+my\s+favorite\s+pizza\s+restaurant\s+(to|as)\s+(.+)$", t, flags=re.IGNORECASE)
    if m_upd_r:
        updates["fav_pizza_restaurant"] = m_upd_r.group(3).strip()

    # Forget commands (delete keys)
    if re.search(r"^forget\s+my\s+favorite\s+pizza\s+restaurant\b", t, flags=re.IGNORECASE):
        deletes.append("fav_pizza_restaurant")
    if re.search(r"^forget\s+my\s+favorite\s+pizza\b", t, flags=re.IGNORECASE) or re.search(r"^forget\s+my\s+pizza\s+preference\b", t, flags=re.IGNORECASE):
        deletes.append("fav_pizza_order")
    if re.search(r"^forget\s+my\s+office\b", t, flags=re.IGNORECASE):
        deletes.append("office_address")
    if re.search(r"^forget\s+my\s+home\b", t, flags=re.IGNORECASE):
        deletes.append("home_address")
    if re.search(r"^forget\s+my\s+diet\b", t, flags=re.IGNORECASE):
        deletes.append("diet")
    if re.search(r"^forget\s+my\s+restrictions\b", t, flags=re.IGNORECASE):
        deletes.append("avoid_items")
    if re.search(r"^forget\s+my\s+ride\s+(app|provider)\b", t, flags=re.IGNORECASE):
        deletes.append("ride_provider")

    return {"updates": updates, "deletes": deletes}

# -------------------------
# Action planning (rides + food)
# -------------------------
def plan_action(user_text: str, mem: Dict[str, str]) -> Optional[Dict[str, Any]]:
    t = user_text.strip().lower()

    # Build restriction list
    avoid_items = parse_csv_list(mem.get("avoid_items"))
    avoid_defaults = infer_default_avoid_from_diet_and_religion(mem)
    avoid_all = list(dict.fromkeys(avoid_items + avoid_defaults))

    # --- BOOK RIDE ---
    if any(phrase in t for phrase in ["book a cab", "book cab", "book a car", "book ride", "book a ride", "get me a cab", "grab me", "book an uber", "book a lyft", "book a didi", "book a bolt"]):
        to = None
        if "to my office" in t or "to office" in t:
            to = mem.get("office_address")
        elif "to my home" in t or "to home" in t:
            to = mem.get("home_address")
        else:
            m = re.search(r"\bto\s+(.+)$", user_text, flags=re.IGNORECASE)
            if m:
                to = m.group(1).strip()

        provider = choose_ride_provider(user_text, mem)

        return {
            "action": "BOOK_RIDE",
            "args": {"provider": provider, "from": "current_location", "to": to},
            "needs_confirmation": True,
            "missing": ["to"] if not to else [],
            "conflicts": []
        }

    # --- ORDER FOOD (generic) ---
    # Triggers for ANY food order
    if re.match(r"^\s*(order|get me|buy me)\s+(.+)$", user_text, flags=re.IGNORECASE):
        # Extract requested item
        item = re.sub(r"^\s*(order|get me|buy me)\s+", "", user_text, flags=re.IGNORECASE).strip()

        # Conflicts using restrictions
        conflicts = contains_any(item, avoid_all)

        return {
            "action": "ORDER_FOOD",
            "args": {
                "provider": "AUTO",  # later: Foodpanda/GrabFood/etc
                "item": item,
                "deliver_to": mem.get("home_address"),
            },
            "needs_confirmation": True,
            "missing": ["deliver_to"] if not mem.get("home_address") else [],
            "conflicts": conflicts
        }

    # --- ORDER PIZZA (specialized convenience) ---
    # Optional: keeps your nice pizza-specific flow when user says pizza explicitly
    if "pizza" in t and any(phrase in t for phrase in ["order", "get me", "buy me"]):
        restaurant = mem.get("fav_pizza_restaurant")
        order = mem.get("fav_pizza_order")
        deliver_to = mem.get("home_address")

        # If user specifies restaurant “from <x>”
        m = re.search(r"\bfrom\s+(.+)$", user_text, flags=re.IGNORECASE)
        if m:
            restaurant = m.group(1).strip()

        conflicts = contains_any(order or "", avoid_all) if order else []

        missing = []
        if not restaurant:
            missing.append("restaurant")
        if not order:
            missing.append("order")

        return {
            "action": "ORDER_PIZZA",
            "args": {"provider": "FOODPANDA", "restaurant": restaurant, "order": order, "deliver_to": deliver_to},
            "needs_confirmation": True,
            "missing": missing,
            "conflicts": conflicts
        }

    return None

# -------------------------
# Endpoints
# -------------------------
@app.get("/health")
def health():
    return {"ok": True, "service": "synq"}

@app.get("/memory/{session_id}")
def read_memory(session_id: str):
    return {"session_id": session_id, "memory": get_memory(session_id)}

@app.post("/chat")
def chat(req: ChatRequest):
    # Load current memory
    mem_before = get_memory(req.session_id)

    # Apply memory deletes/updates
    parsed = extract_memory_updates_and_deletes(req.message, mem_before)
    updates = parsed["updates"]
    deletes = parsed["deletes"]

    memory_changes: List[str] = []

    for key in deletes:
        delete_memory(req.session_id, key)
        memory_changes.append(f"deleted: {key}")

    for k, v in updates.items():
        set_memory(req.session_id, k, v)
        memory_changes.append(f"{k} = {v}")

    # Reload memory after changes
    mem = get_memory(req.session_id)

    # Build plan
    plan = plan_action(req.message, mem)

    # System context
    mem_text = "\n".join([f"- {k}: {v}" for k, v in mem.items()]) if mem else "(none)"

    system = (
        "You are Synq, an action-oriented personal AI agent.\n"
        "You remember user preferences and restrictions and use them to help plan tasks.\n"
        "You DO NOT actually book/pay/order; you only prepare a plan and ask for confirmation.\n"
        "If there is a conflict (requested item violates restrictions), ask to update the preference or choose an alternative.\n\n"
        f"User memory:\n{mem_text}\n"
    )

    # Decide what to say (keep it short + confirmation-first)
    user_msg = req.message

    if plan:
        # Conflicts
        conflicts = plan.get("conflicts") or []
        missing = plan.get("missing") or []

        if conflicts:
            conflict_list = ", ".join(conflicts)
            user_msg = (
                "The user requested something that conflicts with their restrictions.\n"
                f"Conflicting items detected: {conflict_list}.\n"
                "Apologize briefly, ask if they want to change the request or update their preferences, "
                "and offer a vegetarian/allowed alternative."
            )
        elif missing:
            if plan["action"] == "BOOK_RIDE":
                user_msg = "You are planning a ride but the destination is missing. Ask where they want to go (office/home or a place)."
            elif plan["action"] in ["ORDER_FOOD", "ORDER_PIZZA"]:
                user_msg = "You are planning a food order but delivery address or order details are missing. Ask for what’s missing."
            else:
                user_msg = "Ask for the missing details needed to proceed."
        else:
            if plan["action"] == "BOOK_RIDE":
                user_msg = (
                    f"Confirm: plan a {plan['args'].get('provider')} ride to {plan['args'].get('to')}. "
                    "Ask the user to reply YES to confirm or NO to cancel."
                )
            elif plan["action"] == "ORDER_FOOD":
                user_msg = (
                    f"Confirm: order '{plan['args'].get('item')}' and deliver to '{plan['args'].get('deliver_to')}'. "
                    "Ask the user to reply YES to confirm or NO to cancel."
                )
            elif plan["action"] == "ORDER_PIZZA":
                user_msg = (
                    f"Confirm: order pizza via Foodpanda from '{plan['args'].get('restaurant')}' with order '{plan['args'].get('order')}'. "
                    "Ask the user to reply YES to confirm or NO to cancel."
                )

    resp = client.responses.create(
        model=MODEL,
        input=[
            {"role": "system", "content": system},
            {"role": "user", "content": user_msg},
        ],
    )

    return {
        "reply": (resp.output_text or "").strip(),
        "memory_changes": memory_changes,
        "plan": plan
    }
