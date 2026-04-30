"""
LangGraph pipeline:

  retrieve  ->  rerank  ->  compress  ->  generate  ->  verify

We use LangGraph's StateGraph so each step is independently inspectable and
the pipeline can branch in the future (e.g. agentic re-retrieval if
groundedness is low).

Two public entry points:
  - run_pipeline(question, history) -> final state dict (non-streaming)
  - stream_pipeline(question, history) -> iterator of step events + tokens
"""
from __future__ import annotations

from typing import TypedDict, Iterator
from langgraph.graph import StateGraph, START, END
from langchain_core.documents import Document

from .retriever import hybrid_retrieve, rewrite_query
from .reranker import rerank
from .compressor import compress_documents
from .generator import synthesize, stream_answer, format_sources
from .verifier import verify
from .config import SETTINGS


class RAGState(TypedDict, total=False):
    question: str
    history: list[dict]
    rewrites: list[str]
    candidates: list[Document]
    reranked: list[Document]
    compressed: list[Document]
    answer: str
    citations: list[dict]
    verification: dict


# ---------- Nodes ----------
def n_retrieve(state: RAGState) -> RAGState:
    q = state["question"]
    rewrites = rewrite_query(q, SETTINGS.multi_query_n) if SETTINGS.use_multi_query else [q]
    cands = hybrid_retrieve(q)
    return {"rewrites": rewrites, "candidates": cands}


def n_rerank(state: RAGState) -> RAGState:
    return {"reranked": rerank(state["question"], state.get("candidates", []))}


def n_compress(state: RAGState) -> RAGState:
    return {"compressed": compress_documents(state["question"], state.get("reranked", []))}


def n_generate(state: RAGState) -> RAGState:
    answer, citations = synthesize(
        state["question"], state.get("compressed", []), state.get("history", [])
    )
    return {"answer": answer, "citations": citations}


def n_verify(state: RAGState) -> RAGState:
    return {
        "verification": verify(
            state["question"], state.get("answer", ""), state.get("compressed", [])
        )
    }


# ---------- Graph ----------
def build_graph():
    g = StateGraph(RAGState)
    g.add_node("retrieve", n_retrieve)
    g.add_node("rerank", n_rerank)
    g.add_node("compress", n_compress)
    g.add_node("generate", n_generate)
    g.add_node("verify", n_verify)

    g.add_edge(START, "retrieve")
    g.add_edge("retrieve", "rerank")
    g.add_edge("rerank", "compress")
    g.add_edge("compress", "generate")
    g.add_edge("generate", "verify")
    g.add_edge("verify", END)
    return g.compile()


_graph = None
def get_graph():
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph


# ---------- Public API ----------
def run_pipeline(question: str, history: list[dict] | None = None) -> dict:
    return get_graph().invoke({"question": question, "history": history or []})


def stream_pipeline(question: str, history: list[dict] | None = None) -> Iterator[dict]:
    """Yields events:
       {'type': 'step', 'name': 'retrieve', 'detail': '...'}
       {'type': 'token', 'content': '...'}
       {'type': 'final', 'answer': ..., 'citations': ..., 'verification': ...}
    """
    history = history or []

    # Step: retrieve
    yield {"type": "step", "name": "rewrite", "detail": "Rewriting query into multi-query variants…"}
    rewrites = rewrite_query(question, SETTINGS.multi_query_n) if SETTINGS.use_multi_query else [question]
    yield {"type": "step", "name": "rewrite", "detail": f"{len(rewrites)} variants"}

    yield {"type": "step", "name": "retrieve", "detail": "Hybrid retrieval (dense + BM25)…"}
    cands = hybrid_retrieve(question)
    yield {"type": "step", "name": "retrieve", "detail": f"{len(cands)} candidates"}

    # Step: rerank
    yield {"type": "step", "name": "rerank", "detail": "Cross-encoder reranking…"}
    ranked = rerank(question, cands)
    yield {"type": "step", "name": "rerank", "detail": f"top {len(ranked)} retained"}

    # Step: compress
    yield {"type": "step", "name": "compress", "detail": "Extracting only relevant passages…"}
    compressed = compress_documents(question, ranked)
    yield {"type": "step", "name": "compress", "detail": f"{len(compressed)} passages kept"}

    # Step: generate (streaming)
    yield {"type": "step", "name": "generate", "detail": "Synthesizing grounded answer…"}
    full_text = ""
    citations: list[dict] = []
    for token, cits in stream_answer(question, compressed, history):
        citations = cits
        full_text += token
        yield {"type": "token", "content": token}

    # Step: verify
    yield {"type": "step", "name": "verify", "detail": "Self-RAG verification…"}
    v = verify(question, full_text, compressed)
    yield {
        "type": "final",
        "answer": full_text,
        "citations": citations,
        "verification": v,
        "rewrites": rewrites,
    }
