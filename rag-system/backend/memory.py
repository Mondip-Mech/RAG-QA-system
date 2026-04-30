"""
Chat thread storage + semantic search over past conversations.

- Threads are stored as JSON files under data/threads/<thread_id>.json
- We maintain a separate Chroma collection 'thread_messages' for semantic
  search across all messages in all threads
- This is independent of the document KB collection
"""
from __future__ import annotations

import json
import uuid
from datetime import datetime
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document

from .config import SETTINGS, THREADS_DIR
from .ingestion import get_embeddings


THREAD_COLLECTION = "thread_messages"


def _thread_vs() -> Chroma:
    return Chroma(
        collection_name=THREAD_COLLECTION,
        embedding_function=get_embeddings(),
        persist_directory=SETTINGS.chroma_dir,
    )


def new_thread(title: str = "New chat") -> dict:
    tid = uuid.uuid4().hex[:12]
    thread = {
        "id": tid,
        "title": title,
        "created_at": datetime.utcnow().isoformat(),
        "updated_at": datetime.utcnow().isoformat(),
        "messages": [],
    }
    save_thread(thread)
    return thread


def thread_path(tid: str) -> Path:
    return THREADS_DIR / f"{tid}.json"


def save_thread(thread: dict) -> None:
    thread["updated_at"] = datetime.utcnow().isoformat()
    thread_path(thread["id"]).write_text(json.dumps(thread, indent=2))


def load_thread(tid: str) -> dict | None:
    p = thread_path(tid)
    if not p.exists():
        return None
    return json.loads(p.read_text())


def list_threads() -> list[dict]:
    out = []
    for p in sorted(THREADS_DIR.glob("*.json"), key=lambda x: x.stat().st_mtime, reverse=True):
        try:
            t = json.loads(p.read_text())
            out.append({
                "id": t["id"],
                "title": t.get("title", "Untitled"),
                "updated_at": t.get("updated_at"),
                "n_messages": len(t.get("messages", [])),
            })
        except Exception:
            continue
    return out


def delete_thread(tid: str) -> None:
    p = thread_path(tid)
    if p.exists():
        p.unlink()
    # Also remove from the message vector store
    try:
        _thread_vs().delete(where={"thread_id": tid})
    except Exception:
        pass


def append_message(thread: dict, role: str, content: str, citations: list[dict] | None = None) -> None:
    msg = {
        "role": role,
        "content": content,
        "ts": datetime.utcnow().isoformat(),
        "citations": citations or [],
    }
    thread["messages"].append(msg)
    # Auto-title from the first user message
    if role == "user" and thread.get("title", "New chat") == "New chat":
        thread["title"] = content[:60]
    save_thread(thread)

    # Index into semantic search collection (only user + assistant text)
    if content.strip():
        try:
            _thread_vs().add_documents([
                Document(
                    page_content=content,
                    metadata={
                        "thread_id": thread["id"],
                        "thread_title": thread["title"],
                        "role": role,
                        "ts": msg["ts"],
                    },
                )
            ])
        except Exception:
            pass


def search_conversations(query: str, k: int = 5) -> list[dict]:
    try:
        results = _thread_vs().similarity_search(query, k=k)
    except Exception:
        return []
    return [
        {
            "thread_id": r.metadata.get("thread_id"),
            "thread_title": r.metadata.get("thread_title"),
            "role": r.metadata.get("role"),
            "snippet": r.page_content[:240],
            "ts": r.metadata.get("ts"),
        }
        for r in results
    ]
