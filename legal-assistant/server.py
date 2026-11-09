"""Веб-интерфейс чата: FastAPI + статическая страница с просмотром подтверждающих документов."""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fastapi import FastAPI, HTTPException  # noqa: E402
from fastapi.responses import FileResponse  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

from legal_rag import config as cfg  # noqa: E402
from legal_rag import llm  # noqa: E402
from legal_rag.assistant import Turn  # noqa: E402
from legal_rag.runtime import build_assistant  # noqa: E402

STATIC = Path(__file__).resolve().parent / "static"
app = FastAPI(title="Юридический ассистент")
assistant = build_assistant()


class Message(BaseModel):
    role: str = Field(pattern="^(user|assistant)$")
    content: str = Field(max_length=8000)


class ChatRequest(BaseModel):
    messages: list[Message] = Field(min_length=1, max_length=40)


@app.get("/api/health")
def health() -> dict:
    return {
        "status": "ok",
        "documents": len(assistant.kb.documents),
        "chunks": len(assistant.kb.chunks),
        "embeddings": assistant.searcher.vectors is not None,
        "llm_configured": bool(cfg.API_BASE and cfg.LLM_KEY),
        "model": cfg.LLM_MODEL,
    }


@app.get("/api/documents")
def documents() -> list[dict]:
    return [{"doc_id": d.doc_id, "title": d.title, "category": d.category, "status": d.status,
             "chars": len(d.text)} for d in assistant.kb.documents]


@app.get("/api/documents/{doc_id}")
def document(doc_id: str) -> dict:
    doc = assistant.kb.by_id.get(doc_id)
    if not doc:
        raise HTTPException(404, "Документ не найден")
    return {"doc_id": doc.doc_id, "title": doc.title, "category": doc.category, "status": doc.status,
            "text": doc.text, "removed_instructions": len(doc.removed)}


@app.post("/api/chat")
def chat(req: ChatRequest) -> dict:
    turns = [Turn(m.role, m.content.strip()) for m in req.messages if m.content.strip()]
    if not turns or turns[-1].role != "user":
        raise HTTPException(400, "Последнее сообщение должно быть от пользователя")
    started = time.time()
    with llm.make_client() as client:
        result = assistant.answer(client, turns)
    return {
        "answer": result.text,
        "sources": result.sources,
        "documents": result.documents,
        "verified": result.verified,
        "analysis": result.analysis,
        "error": result.error,
        "seconds": round(time.time() - started, 1),
    }


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC / "index.html")


app.mount("/static", StaticFiles(directory=STATIC), name="static")
