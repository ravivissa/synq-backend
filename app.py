import os
from fastapi import FastAPI
from pydantic import BaseModel
from openai import OpenAI

app = FastAPI(title="Synq Backend")
client = OpenAI()

MODEL = "gpt-4o-mini"  # forced model

class ChatRequest(BaseModel):
    session_id: str
    message: str

@app.get("/health")
def health():
    return {"ok": True, "service": "synq"}

@app.post("/chat")
def chat(req: ChatRequest):
    # Using Responses API (simple text in, text out)
    resp = client.responses.create(
        model=MODEL,
        input=req.message
    )
    return {"reply": (resp.output_text or "").strip()}





