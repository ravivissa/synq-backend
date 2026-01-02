import os
import re
from typing import Dict, Optional, Any, List

from fastapi import FastAPI
from pydantic import BaseModel
from openai import OpenAI

import psycopg

# Optional: only needed if you use /execute to send to Make.com
try:
    import requests
except Exception:
    requests = None


# -------------------------
# App setup
# -------------------------
app = FastAPI(title="Synq Backend")
client = OpenAI()

MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
DATABASE_URL = os.getenv("DATABASE_URL")
MAKE_WEBHOOK_URL = os.getenv("MAKE_WEBHOOK_URL")  # optional

if not DATABASE_URL:
    raise RuntimeError(
        "DATABASE_URL is not set. In Railway: add Postgres and reference DATABASE_URL into synq-backend variables."
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


class RankFoodRequest(BaseModel):
    session_id: str
    item: str
    service: str = "AUTO"
    options: list  # list of dicts from Make.com (name, distance_km, price_level, rating, etc.)


class ExecuteRequest(BaseModel):
    session_id: str
    plan: dict
    user_confirmed: bool = True


# -------------------------
# Helpers
# -------------------------
TOP_RIDE_PROVIDERS = ["UBER", "DIDI", "LYFT", "GRAB", "BOLT"]
KNOWN_FOOD_SERVICES = ["FOODPANDA", "GRABFOOD", "UBEREATS", "DOORDASH", "SWIGGY", "ZOMATO", "DELIVEROO", "AUTO"]


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


def normalize_food_service(name: Optional[str]) -> Optional[str]:
    if not name:
        return None
    n = name.strip().lower().replace(" ", "")
    mapping = {
        "foodpanda": "FOODPANDA",
        "grabfood": "GRABFOOD",
        "uber": "UBEREATS",
        "ubereats": "UBEREATS",
        "doordash": "DOORDASH",
        "swiggy": "SWIGGY",
        "zomato": "ZOMATO",
        "deliveroo": "DELIVEROO",
        "auto": "AUTO",
    }
    return mapping.get(n, name.strip().upper().replace(" ", "_"))


def parse_csv_list(s: Optional[str]) -> List[str]:
    if not s:
        return []
    return [x.strip().lower() for x in s.split(",") if x.strip()]


def add_csv_item(existing_csv: Optional[str], item: str) -> str:
    existing = set(parse_csv_list(existing_csv))
    it = (item or "").strip().lower()
    if it:
        existing.add(it)
    return ", ".join(sorted(existing))


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

    # Diet defaults
    if diet in ["vegetarian", "vegan"]:
        defaults += ["pepperoni", "pork", "bacon", "ham", "beef", "chicken", "sausage", "salami", "meat", "fish", "shrimp"]
    if diet == "vegan":
        defaults += ["cheese", "mozzarella", "milk", "butter", "cream", "yogurt", "paneer", "egg", "eggs"]

    # Conservative religion defaults (ONLY if user explicitly said religion)
    if religion == "hindu":
        defaults += ["beef"]
    if religion == "muslim":
        defaults += ["pork", "bacon", "ham"]
    if religion == "jain":
        defaults += ["onion", "garlic", "potato"]

    # Unique preserve order
    seen = set()
    out = []
    for d in defaults:
        dl = d.lower()
        if dl not in seen:
            seen.add(dl)
            out.append(dl)
    return out


def choose_ride_provider(user_text: str, mem: Dict[str, str]) -> str:
    lt = user_text.lower()
    for p in ["uber", "didi", "lyft", "grab", "bolt"]:
        if p in lt:
            return normalize_provider(p) or "UBER"

    pref = normalize_provider(mem.get("ride_provider"))
    if pref and pref in TOP_RIDE_PROVIDERS:
        return pref

    return "UBER"


# -------------------------
# Memory parsing (updates + deletes)
# -------------------------
def extract_memory_changes(text: str, mem: Dict[str, str]) -> Dict[str, Any]:
    """
    Returns {"updates": {...}, "deletes": [...]}
    """
    t = text.strip()
    updates: Dict[str, str] = {}
    deletes: List[str] = []

    # Locations
    m = re.match(r"^remember\s+my\s+office\s+is\s+(.+)$", t, flags=re.IGNORECASE)
    if m:
        updates["office_address"] = m.group(1).strip()

    m = re.match(r"^remember\s+my\s+home\s+is\s+(.+)$", t, flags=re.IGNORECASE)
    if m:
        updates["home_address"] = m.group(1).strip()

    # Ride provider preference
    m = re.match(r"^remember\s+my\s+ride\s+(app|provider)\s+is\s+(.+)$", t, flags=re.IGNORECASE)
    if m:
        prov = normalize_provider(m.group(2))
        if prov in TOP_RIDE_PROVIDERS or prov == "AUTO":
            updates["ride_provider"] = prov

    # Food preferences (general)
    # "Remember my favorite cuisines are burgers, indian"
    m = re.match(r"^remember\s+my\s+favorite\s+cuisines?\s+(are|is)\s+(.+)$", t, flags=re.IGNORECASE)
    if m:
        updates["fav_cuisines"] = m.group(2).strip().lower()

    # "Remember my favorite restaurants are X, Y"
    m = re.match(r"^remember\s+my\s+favorite\s+restaurants?\s+(are|is)\s+(.+)$", t, flags=re.IGNORECASE)
    if m:
        updates["fav_restaurants"] = m.group(2).strip().lower()

    # Price sensitivity
    m = re.match(r"^remember\s+my\s+price\s+sensitivity\s+is\s+(low|medium|high)$", t, flags=re.IGNORECASE)
    if m:
        updates["price_sensitivity"] = m.group(1).strip().lower()

    # Diet
    if re.search(r"\b(i am|i'm)\s+vegetarian\b", t, flags=re.IGNORECASE):
        updates["diet"] = "vegetarian"
    if re.search(r"\b(i am|i'm)\s+vegan\b", t, flags=re.IGNORECASE):
        updates["diet"] = "vegan"
    if re.search(r"\b(i eat\s+non-veg|i am\s+non-veg|i eat meat)\b", t, flags=re.IGNORECASE):
        updates["diet"] = "non_veg"

    # Religion (only if user says it)
    m = re.search(r"\b(i am|i'm)\s+(hindu|muslim|christian|jain|sikh|buddhist)\b", t, flags=re.IGNORECASE)
    if m:
        updates["religion"] = m.group(2).strip().lower()

    # Avoid items: "I don't eat X" / "I do not eat X"
    m = re.search(r"\b(i (?:do not|don't) eat)\s+(.+)$", t, flags=re.IGNORECASE)
    if m:
        item = m.group(2).strip().rstrip(".")
        if len(item) > 60:
            item = item[:60].strip()
        updates["avoid_items"] = add_avoid_items(mem.get("avoid_items"), [item])

    # Add to favorite restaurants automatically (optional commands)
    m = re.match(r"^add\s+(.+)\s+to\s+my\s+favorite\s+restaurants$", t, flags=re.IGNORECASE)
    if m:
        updates["fav_restaurants"] = add_csv_item(mem.get("fav_restaurants"), m.group(1).strip())

    # Forget commands
    if re.match(r"^forget\s+my\s+office$", t, flags=re.IGNORECASE):
        deletes.append("office_address")
    if re.match(r"^forget\s+my\s+home$", t, flags=re.IGNORECASE):
        deletes.append("home_address")
    if re.match(r"^forget\s+my\s+diet$", t, flags=re.IGNORECASE):
        deletes.append("diet")
    if re.match(r"^forget\s+my\s+restrictions$", t, flags=re.IGNORECASE):
        deletes.append("avoid_items")
    if re.match(r"^forget\s+my\s+ride\s+(app|provider)$", t, flags=re.IGNORECASE):
        deletes.append("ride_provider")
    if re.match(r"^forget\s+my\s+favorite\s+restaurants?$", t, flags=re.IGNORECASE):
        deletes.append("fav_restaurants")
    if re.match(r"^forget\s+my\s+favorite\s+cuisines?$", t, flags=re.IGNORECASE):
        deletes.append("fav_cuisines")
    if re.match(r"^forget\s+my\s+price\s+sensitivity$", t, flags=re.IGNORECASE):
        deletes.append("price_sensitivity")

    return {"updates": updates, "deletes": deletes}


# -------------------------
# Action planning
# -------------------------
def plan_action(user_text: str, mem: Dict[str, str]) -> Optional[Dict[str, Any]]:
    t = user_text.strip().lower()

    # Restrictions list for ALL food checks
    avoid_items = parse_csv_list(mem.get("avoid_items"))
    avoid_defaults = infer_default_avoid_from_diet_and_religion(mem)
    avoid_all = list(dict.fromkeys(avoid_items + avoid_defaults))

    # ---- BOOK RIDE (top 5 providers) ----
    if any(p in t for p in ["book a cab", "book cab", "book a ride", "book ride", "get me a cab", "get me a ride", "book an uber", "book a grab", "book a lyft", "book a didi", "book a bolt"]):
        dest = None
        if "to my office" in t or "to office" in t:
            dest = mem.get("office_address")
        elif "to my home" in t or "to home" in t:
            dest = mem.get("home_address")
        else:
            m = re.search(r"\bto\s+(.+)$", user_text, flags=re.IGNORECASE)
            if m:
                dest = m.group(1).strip()

        provider = choose_ride_provider(user_text, mem)
        return {
            "action": "BOOK_RIDE",
            "args": {"provider": provider, "from": "current_location", "to": dest},
            "needs_confirmation": True,
            "missing": ["to"] if not dest else [],
            "conflicts": []
        }

    # ---- ORDER FOOD (specific or generic) ----
    # supports:
    # "Order chicken burger"
    # "Order chicken burger from McDonald's"
    # "Order chicken burger from McDonald's on Foodpanda"
    m = re.match(r"^\s*(order|get me|buy me)\s+(.+)$", user_text, flags=re.IGNORECASE)
    if m:
        raw = re.sub(r"^\s*(order|get me|buy me)\s+", "", user_text, flags=re.IGNORECASE).strip()

        # parse "on/via <service>"
        service = None
        ms = re.search(r"\b(on|via)\s+([a-zA-Z0-9_\- ]+)$", raw, flags=re.IGNORECASE)
        if ms:
            service = normalize_food_service(ms.group(2).strip())
            raw = raw[: ms.start()].strip()

        # parse "from <restaurant>"
        restaurant = None
        mr = re.search(r"\bfrom\s+(.+)$", raw, flags=re.IGNORECASE)
        if mr:
            restaurant = mr.group(1).strip()
            raw = raw[: mr.start()].strip()

        item = raw.strip()

        # If request is NOT specific (no restaurant and no service), we go to search mode
        # because user asked: "give me 5 options by proximity/price/preferences"
        if not restaurant and not service:
            conflicts = contains_any(item, avoid_all)
            return {
                "action": "SEARCH_FOOD_OPTIONS",
                "args": {
                    "service": "AUTO",
                    "item": item,
                    "near": "current_location",
                    "max_results": 20
                },
                "needs_confirmation": False,
                "missing": ["item"] if not item else [],
                "conflicts": conflicts
            }

        # Specific order -> confirm then execute after YES
        deliver_to = mem.get("home_address")
        conflicts = contains_any(item, avoid_all)

        missing = []
        if not item:
            missing.append("item")
        if not deliver_to:
            missing.append("deliver_to")

        return {
            "action": "ORDER_FOOD",
            "args": {
                "service": (service or "AUTO"),
                "restaurant": restaurant,
                "item": item,
                "deliver_to": deliver_to
            },
            "needs_confirmation": True,
            "missing": missing,
            "conflicts": conflicts
        }

    return None


# -------------------------
# Ranking (top 5 options)
# Make.com calls this after it fetches candidates.
# -------------------------
@app.post("/rank_food_options")
def rank_food_options(req: RankFoodRequest):
    mem = get_memory(req.session_id)

    fav_restaurants = set(parse_csv_list(mem.get("fav_restaurants")))
    fav_cuisines = set(parse_csv_list(mem.get("fav_cuisines")))
    price_pref = (mem.get("price_sensitivity") or "medium").lower()

    def price_score(price_level: Optional[int]) -> float:
        # price_level: 1 cheap ... 4 expensive (common in Google Places style)
        if price_level is None:
            return 0.5
        pl = max(1, min(4, int(price_level)))
        if price_pref == "low":
            return 1.0 - (pl - 1) * 0.25
        if price_pref == "high":
            return 0.25 + (pl - 1) * 0.25
        return 0.75 - (pl - 1) * 0.15  # medium

    ranked = []
    for opt in req.options:
        name = (opt.get("name") or "").strip()
        name_l = name.lower()
        cuisine = (opt.get("cuisine") or "").strip().lower()

        distance_km = float(opt.get("distance_km") or 999)
        price_level = opt.get("price_level")  # 1..4 if present
        rating = float(opt.get("rating") or 0)

        # Normalize signals
        s_distance = max(0.0, 1.0 - (distance_km / 10.0))  # closer than 10km gets >0
        s_price = price_score(price_level)
        s_rating = min(1.0, rating / 5.0)

        # Preference boost
        s_pref = 0.0
        if name_l in fav_restaurants:
            s_pref += 0.5
        if cuisine and cuisine in fav_cuisines:
            s_pref += 0.25

        # Weighted score: proximity + price + rating + prefs
        score = (0.45 * s_distance) + (0.25 * s_price) + (0.20 * s_rating) + (0.10 * s_pref)

        ranked.append({**opt, "score": round(score, 4)})

    ranked.sort(key=lambda x: x["score"], reverse=True)
    top5 = ranked[:5]

    return {"item": req.item, "service": req.service, "top_5": top5}


# -------------------------
# Execute: forward plan to Make.com webhook (optional)
# Call this only after user says YES in Voiceflow.
# -------------------------
@app.post("/execute")
def execute(req: ExecuteRequest):
    if not req.user_confirmed:
        return {"status": "cancelled", "message": "Okay, cancelled."}

    if not MAKE_WEBHOOK_URL:
        return {"status": "error", "message": "MAKE_WEBHOOK_URL is not configured in Railway variables."}

    if requests is None:
        return {"status": "error", "message": "requests not installed. Add 'requests' to requirements.txt if using /execute."}

    payload = {
        "session_id": req.session_id,
        "plan": req.plan
    }

    r = requests.post(MAKE_WEBHOOK_URL, json=payload, timeout=15)
    return {"status": "sent_to_make", "make_status": r.status_code, "make_response": r.text}


# -------------------------
# Core endpoints
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

    # Apply memory changes
    parsed = extract_memory_changes(req.message, mem_before)
    updates = parsed["updates"]
    deletes = parsed["deletes"]

    memory_changes: List[str] = []

    for k in deletes:
        delete_memory(req.session_id, k)
        memory_changes.append(f"deleted: {k}")

    for k, v in updates.items():
        set_memory(req.session_id, k, v)
        memory_changes.append(f"{k} = {v}")

    # Reload memory
    mem = get_memory(req.session_id)

    # Create plan
    plan = plan_action(req.message, mem)

    # Build a compact system context
    mem_lines = "\n".join([f"- {k}: {v}" for k, v in mem.items()]) if mem else "(none)"
    system = (
        "You are Synq, an action-oriented personal AI agent.\n"
        "You remember user preferences and restrictions.\n"
        "You do NOT actually book or pay — you create plans and ask for confirmation.\n"
        "If there is a conflict with dietary restrictions, ask to adjust the request.\n"
        "Be concise.\n\n"
        f"User memory:\n{mem_lines}\n"
    )

    # Decide what message to feed the model
    user_msg = req.message

    if plan:
        conflicts = plan.get("conflicts") or []
        missing = plan.get("missing") or []

        if conflicts:
            user_msg = (
                f"User requested: '{req.message}'.\n"
                f"Conflict items detected: {', '.join(conflicts)}.\n"
                "Apologize briefly, ask if they want to change the item to something allowed, "
                "and offer to remember the updated preference."
            )
        elif missing:
            # Ask only what’s missing
            if plan["action"] == "BOOK_RIDE" and "to" in missing:
                user_msg = "Ask where they want to go (office/home or a specific place)."
            elif plan["action"] == "ORDER_FOOD" and "deliver_to" in missing:
                user_msg = "Ask for their delivery address (or ask them to save home address)."
            elif plan["action"] == "SEARCH_FOOD_OPTIONS" and "item" in missing:
                user_msg = "Ask what food item they want."
            else:
                user_msg = "Ask for the missing details needed to proceed."
        else:
            # Confirmation prompt for specific actions
            if plan["action"] == "BOOK_RIDE":
                user_msg = (
                    f"Confirm: plan a {plan['args'].get('provider')} ride to {plan['args'].get('to')}. "
                    "Tell them to reply YES to confirm or NO to cancel."
                )
            elif plan["action"] == "ORDER_FOOD":
                user_msg = (
                    f"Confirm: order '{plan['args'].get('item')}' from '{plan['args'].get('restaurant')}' "
                    f"on '{plan['args'].get('service')}', deliver to '{plan['args'].get('deliver_to')}'. "
                    "Tell them to reply YES to confirm or NO to cancel."
                )
            elif plan["action"] == "SEARCH_FOOD_OPTIONS":
                user_msg = (
                    f"User wants '{plan['args'].get('item')}'. "
                    "Say you will show 5 options based on proximity, price, and their preferences."
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
