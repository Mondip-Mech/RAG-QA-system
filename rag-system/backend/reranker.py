"""
Cross-encoder reranking with BAAI/bge-reranker-base.

Cross-encoders score (query, passage) pairs jointly and consistently outperform
bi-encoder scores for relevance ranking — at the cost of being slower. We only
run the reranker on the top-N candidates from hybrid retrieval, never on the
full corpus.

Loaded lazily so the module import is cheap; the model is ~280MB.
"""
from __future__ import annotations

from typing import Optional
from langchain_core.documents import Document

from .config import SETTINGS


_model = None


def _load_model():
    global _model
    if _model is not None:
        return _model
    try:
        from sentence_transformers import CrossEncoder
        _model = CrossEncoder(SETTINGS.reranker_model)
    except Exception as e:
        print(f"[reranker] could not load model: {e}")
        _model = False  # sentinel: tried and failed
    return _model


def rerank(query: str, docs: list[Document], top_k: Optional[int] = None) -> list[Document]:
    """Return docs reranked by cross-encoder. Falls back to identity on failure."""
    if not SETTINGS.use_reranker or not docs:
        return docs[: top_k or SETTINGS.top_k_rerank]

    model = _load_model()
    if not model:
        return docs[: top_k or SETTINGS.top_k_rerank]

    pairs = [(query, d.page_content) for d in docs]
    try:
        scores = model.predict(pairs)
    except Exception:
        return docs[: top_k or SETTINGS.top_k_rerank]

    scored = sorted(zip(docs, scores), key=lambda x: -float(x[1]))
    k = top_k or SETTINGS.top_k_rerank
    return [d for d, _ in scored[:k]]
