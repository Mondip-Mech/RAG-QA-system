"""
Optional FastAPI server. Use this if you swap Streamlit for React/Next.js.
Run with:  uvicorn backend.api:app --reload --port 8000

Endpoints:
  POST /upload           - upload one PDF
  POST /sync             - reindex all uploads
  GET  /files            - list indexed files
  DELETE /files/{name}   - remove a file from the KB
  POST /clear            - wipe the KB
  POST /chat             - non-streaming Q&A
  POST /chat/stream      - SSE streaming Q&A with step events
  GET  /threads          - list chat threads
  GET  /threads/{id}     - load thread
  POST /threads          - create thread
  DELETE /threads/{id}   - delete thread
  GET  /search/conversations?q=... - semantic search across threads
"""
from __future__ import annotations

import json
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from .config import SETTINGS, UPLOAD_DIR
from .ingestion import (
    ingest_file,
    ingest_all_uploads,
    list_indexed_files,
    remove_file,
    clear_index,
)
from .graph import run_pipeline, stream_pipeline
from .memory import (
    new_thread,
    load_thread,
    save_thread,
    list_threads,
    delete_thread,
    append_message,
    search_conversations,
)


app = FastAPI(title="Local RAG Document Q&A")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    question: str
    thread_id: str | None = None


@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Only PDF files are supported.")
    dest = UPLOAD_DIR / file.filename
    dest.write_bytes(await file.read())
    res = ingest_file(dest)
    return res


@app.post("/sync")
def sync():
    return {"results": ingest_all_uploads()}


@app.get("/files")
def files():
    return {"files": list_indexed_files()}


@app.delete("/files/{name}")
def files_delete(name: str):
    remove_file(name)
    return {"ok": True}


@app.post("/clear")
def clear():
    clear_index()
    return {"ok": True}


@app.get("/runtime")
def runtime():
    return {
        "llm": SETTINGS.llm_model,
        "embeddings": SETTINGS.embedding_model,
        "vector_db": "ChromaDB",
        "hybrid": SETTINGS.use_hybrid,
        "top_k_rerank": SETTINGS.top_k_rerank,
        "reranker": SETTINGS.reranker_model if SETTINGS.use_reranker else None,
    }


@app.post("/chat")
def chat(req: ChatRequest):
    thread = load_thread(req.thread_id) if req.thread_id else new_thread()
    if thread is None:
        thread = new_thread()
    history = thread["messages"]
    append_message(thread, "user", req.question)
    state = run_pipeline(req.question, history=history)
    append_message(thread, "assistant", state.get("answer", ""), state.get("citations", []))
    return {
        "thread_id": thread["id"],
        "answer": state.get("answer"),
        "citations": state.get("citations", []),
        "verification": state.get("verification", {}),
    }


@app.post("/chat/stream")
def chat_stream(req: ChatRequest):
    thread = load_thread(req.thread_id) if req.thread_id else new_thread()
    if thread is None:
        thread = new_thread()
    history = list(thread["messages"])
    append_message(thread, "user", req.question)

    def gen():
        full = ""
        citations = []
        verification = {}
        for ev in stream_pipeline(req.question, history=history):
            if ev["type"] == "token":
                full += ev["content"]
            if ev["type"] == "final":
                citations = ev.get("citations", [])
                verification = ev.get("verification", {})
            yield f"data: {json.dumps(ev)}\n\n"
        # Persist final assistant message after stream ends
        append_message(thread, "assistant", full, citations)
        yield f"data: {json.dumps({'type':'thread', 'thread_id': thread['id']})}\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream")


@app.get("/threads")
def threads():
    return {"threads": list_threads()}


@app.get("/threads/{tid}")
def threads_get(tid: str):
    t = load_thread(tid)
    if t is None:
        raise HTTPException(404, "Thread not found")
    return t


@app.post("/threads")
def threads_new():
    return new_thread()


@app.delete("/threads/{tid}")
def threads_delete(tid: str):
    delete_thread(tid)
    return {"ok": True}


@app.get("/search/conversations")
def search_convos(q: str, k: int = 5):
    return {"results": search_conversations(q, k=k)}
