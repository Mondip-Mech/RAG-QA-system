"""
Hybrid retrieval with multi-query expansion + RRF fusion.

Performance optimizations:
  - Cloud LLM for query rewriting (fast, free on Groq/NVIDIA)
  - LRU cache on rewrites: identical questions skip the LLM entirely
  - Parallel dense + sparse retrieval across all query variants
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

from .config import SETTINGS
from .ingestion import get_vectorstore, load_bm25
from .llm import get_rewrite_llm


# ---------- Rewrite LLM (lazy singleton) ----------
_rewrite_llm = None
def _get_rewrite_llm():
    global _rewrite_llm
    if _rewrite_llm is None:
        _rewrite_llm = get_rewrite_llm()
    return _rewrite_llm


# ---------- Multi-query rewriting ----------
MULTI_QUERY_PROMPT = ChatPromptTemplate.from_template(
    """You are an AI search assistant. Generate {n} different versions of the
user's question to retrieve relevant documents from a vector database. Each
version should explore a different angle: synonyms, broader/narrower scope,
related concepts.

Return ONLY the questions, one per line. No numbering, no explanations.

Question: {question}"""
)


@lru_cache(maxsize=256)
def _cached_rewrite(question: str, n: int) -> tuple[str, ...]:
    """Cached LLM rewrites. Returns a tuple so it's hashable."""
    try:
        msg = MULTI_QUERY_PROMPT.format_messages(n=n, question=question)
        out = _get_rewrite_llm().invoke(msg).content.strip()
        rewrites = [line.strip(" -*0123456789.") for line in out.splitlines() if line.strip()]
        rewrites = [r for r in rewrites if len(r) > 4][:n]
        return tuple(rewrites)
    except Exception:
        return tuple()


def rewrite_query(question: str, n: int = 3) -> list[str]:
    """Generate N paraphrases. Always includes the original question first."""
    if not SETTINGS.use_multi_query or n <= 1:
        return [question]
    rewrites = list(_cached_rewrite(question, n))
    if question not in rewrites:
        rewrites = [question, *rewrites]
    return rewrites[: n + 1]


# ---------- Reciprocal Rank Fusion ----------
def rrf_fuse(ranked_lists: list[list[Document]], k: int = SETTINGS.rrf_k) -> list[Document]:
    scores: dict[str, float] = {}
    docs_by_key: dict[str, Document] = {}
    for ranked in ranked_lists:
        for rank, doc in enumerate(ranked):
            key = doc.metadata.get("chunk_id") or doc.page_content[:80]
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank + 1)
            if key not in docs_by_key:
                docs_by_key[key] = doc
    return sorted(
        docs_by_key.values(),
        key=lambda d: -scores[d.metadata.get("chunk_id") or d.page_content[:80]],
    )


# ---------- Search primitives ----------
def dense_search(query: str, k: int) -> list[Document]:
    try:
        return get_vectorstore().similarity_search(query, k=k)
    except Exception:
        return []


def sparse_search(query: str, k: int) -> list[Document]:
    bm25 = load_bm25()
    if bm25 is None:
        return []
    bm25.k = k
    try:
        return bm25.invoke(query)
    except Exception:
        return []


# ---------- Hybrid retrieval (parallel) ----------
def hybrid_retrieve(question: str) -> list[Document]:
    """
    Full pipeline:
      1. Cached LLM rewrite into N variants
      2. Parallel dense + sparse search across ALL variants
      3. RRF fuse
    """
    queries = (
        rewrite_query(question, SETTINGS.multi_query_n)
        if SETTINGS.use_multi_query
        else [question]
    )

    jobs: list[tuple[str, str, int]] = []
    for q in queries:
        jobs.append(("dense", q, SETTINGS.top_k_dense))
        if SETTINGS.use_hybrid:
            jobs.append(("sparse", q, SETTINGS.top_k_sparse))

    def run_job(job):
        kind, q, k = job
        return dense_search(q, k) if kind == "dense" else sparse_search(q, k)

    with ThreadPoolExecutor(max_workers=min(8, len(jobs))) as ex:
        ranked_lists = list(ex.map(run_job, jobs))

    fused = rrf_fuse([r for r in ranked_lists if r])
    return fused[: max(SETTINGS.top_k_dense + SETTINGS.top_k_sparse, 12)]