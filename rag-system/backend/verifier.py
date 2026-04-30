"""
Self-RAG style verification — scores answer groundedness and relevance.
"""
from __future__ import annotations

import json
import re
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

from .config import SETTINGS
from .llm import get_chat_llm


_llm = None
def _get_llm():
    global _llm
    if _llm is None:
        _llm = get_chat_llm(temperature=0.0)
    return _llm


VERIFY_PROMPT = ChatPromptTemplate.from_template(
    """You are an answer auditor. Given the question, the proposed answer, and
the sources used, return a JSON object with this schema:
{{
  "groundedness": <0.0-1.0>,
  "relevance":    <0.0-1.0>,
  "issues": ["<short string>", ...]
}}

Return ONLY valid JSON, nothing else.

Question: {question}

Answer:
\"\"\"
{answer}
\"\"\"

Sources:
{sources}
"""
)


def verify(question: str, answer: str, docs: list[Document]) -> dict:
    if not SETTINGS.use_verification or not docs:
        return {"groundedness": 1.0, "relevance": 1.0, "issues": [], "skipped": True}

    sources = "\n".join(f"[S{i+1}] {d.page_content[:600]}" for i, d in enumerate(docs))
    msg = VERIFY_PROMPT.format_messages(question=question, answer=answer, sources=sources)
    try:
        raw = _get_llm().invoke(msg).content
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        return json.loads(m.group(0) if m else raw)
    except Exception as e:
        return {"groundedness": None, "relevance": None, "issues": [f"verifier error: {e}"]}