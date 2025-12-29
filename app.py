import os
from typing import Optional, Dict, Any
from fastapi import FastAPI
from pydantic import BaseModel
from openai import OpenAI



@app.get("/debug/env")
def debug_env():
    return {
        "has_openai_key": bool(os.getenv("OPENAI_API_KEY")),
        "model": os.getenv("OPENAI_MODEL", "not-set"),
    }

# Synq - minimal agent API for Voiceflow
# Endpoints:
#   GET  /health
#   POST /chat

app = FastAPI(title="Synq Agent API")
client = OpenAI()

MODEL = os.getenv("OPENAI_MODEL", "gpt-5")  # you can change later

class ChatRequest(BaseModel):
    session_id: str
    message: str
    mode: Optional[str] = "general"
    memory: Optional[Dict[str, Any]] = None  # optional user memory from Voiceflow

class ChatResponse(BaseModel):
    reply: str

SYSTEM = """
You are Synq, an action-oriented personal AI agent.
You help users get things done across apps and remember preferences.

Behaviors:
- Be concise and confirm before any irreversible action (booking/payment).
- If info is missing (home/office address, pizza preference), ask only what's needed.
- You may propose "next steps" to connect tools (Grab/Foodpanda) via APIs/automation.

If user asks you to remember something, summarize it in 1 line and tag it as MEMORY_SUGGESTION.
"""

@app.get("/health")
def health():
    return {"ok": True, "service": "synq"}

@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    # Inject any memory passed from Voiceflow (optional)
    memory_block = ""
    if req.memory:
        memory_block = "USER MEMORY (trusted):\n" + "\n".join(
            f"- {k}: {v}" for k, v in req.memory.items() if v
        )

    messages = [
        {"role": "system", "content": SYSTEM.strip()},
    ]
    if memory_block:
        messages.append({"role": "system", "content": memory_block})
    messages.append({"role": "user", "content": req.message})

    # Responses API
    resp = client.responses.create(
        model=MODEL,
        input=messages,
    )

    text = (resp.output_text or "").strip()
    if not text:
        text = "I couldn’t generate a reply just now. Please try again."

    return ChatResponse(reply=text)

