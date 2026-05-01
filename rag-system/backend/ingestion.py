"""
Document ingestion pipeline.

- Loads PDFs (PyMuPDF for speed + better text extraction than PyPDF)
- Performs semantic chunking when possible, falls back to recursive splitter
- Persists chunks into Chroma with metadata (source, page, chunk_id)
- Maintains a parallel BM25 index (rebuilt on each sync)
"""
from __future__ import annotations

import hashlib
import json
import pickle
from pathlib import Path
from typing import Iterable

from langchain_core.documents import Document
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever

from .config import SETTINGS, UPLOAD_DIR, CHROMA_DIR


BM25_PICKLE = Path(SETTINGS.chroma_dir) / "bm25.pkl"
MANIFEST_PATH = Path(SETTINGS.chroma_dir) / "manifest.json"


# ---------- Embeddings (singleton-ish, in-process via sentence-transformers) ----------
_embeddings = None
def get_embeddings() -> HuggingFaceEmbeddings:
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(
            model_name=SETTINGS.embedding_model,
            model_kwargs={"device": "cpu"},
            # batch_size 64 ≈ 2× faster on CPU than the default 32 for bge-small;
            # show_progress_bar on stderr makes long jobs visible in the terminal.
            encode_kwargs={
                "normalize_embeddings": True,
                "batch_size": 64,
                "show_progress_bar": True,
            },
        )
    return _embeddings


# ---------- Vector store ----------
def get_vectorstore() -> Chroma:
    return Chroma(
        collection_name=SETTINGS.chroma_collection,
        embedding_function=get_embeddings(),
        persist_directory=SETTINGS.chroma_dir,
    )


# ---------- Manifest (tracks ingested files) ----------
def load_manifest() -> dict:
    if MANIFEST_PATH.exists():
        return json.loads(MANIFEST_PATH.read_text())
    return {"files": {}}


def save_manifest(m: dict) -> None:
    MANIFEST_PATH.write_text(json.dumps(m, indent=2))


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


# ---------- Loading & chunking ----------
def load_pdf(path: Path) -> list[Document]:
    loader = PyMuPDFLoader(str(path))
    docs = loader.load()
    # Normalize metadata
    for d in docs:
        d.metadata["source"] = path.name
        d.metadata["source_path"] = str(path)
        d.metadata["page"] = int(d.metadata.get("page", 0)) + 1  # 1-indexed
    return docs


def make_splitter():
    if SETTINGS.use_semantic_chunking:
        try:
            return SemanticChunker(
                embeddings=get_embeddings(),
                breakpoint_threshold_type="percentile",
                breakpoint_threshold_amount=SETTINGS.semantic_breakpoint_threshold,
            )
        except Exception:
            pass  # fall through to recursive
    return RecursiveCharacterTextSplitter(
        chunk_size=SETTINGS.chunk_size,
        chunk_overlap=SETTINGS.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )


def chunk_documents(docs: list[Document]) -> list[Document]:
    splitter = make_splitter()
    chunks: list[Document] = []
    for i, doc in enumerate(splitter.split_documents(docs)):
        cid = f"{doc.metadata.get('source','?')}::p{doc.metadata.get('page','?')}::c{i}"
        doc.metadata["chunk_id"] = cid
        chunks.append(doc)
    return chunks


# ---------- BM25 (sparse) ----------
def rebuild_bm25(all_docs: Iterable[Document]) -> None:
    docs = list(all_docs)
    if not docs:
        if BM25_PICKLE.exists():
            BM25_PICKLE.unlink()
        return
    bm25 = BM25Retriever.from_documents(docs)
    bm25.k = SETTINGS.top_k_sparse
    with open(BM25_PICKLE, "wb") as f:
        pickle.dump(docs, f)


def load_bm25() -> BM25Retriever | None:
    if not BM25_PICKLE.exists():
        return None
    with open(BM25_PICKLE, "rb") as f:
        docs = pickle.load(f)
    bm25 = BM25Retriever.from_documents(docs)
    bm25.k = SETTINGS.top_k_sparse
    return bm25


# ---------- Public API ----------
def ingest_file(path: Path, progress_cb=None) -> dict:
    """Ingest a single PDF. Skips if already ingested (by hash).

    progress_cb: optional callable(str) — phase messages for UI status updates.
    """
    def _p(msg: str) -> None:
        if progress_cb:
            try:
                progress_cb(msg)
            except Exception:
                pass

    manifest = load_manifest()
    fhash = file_hash(path)
    if path.name in manifest["files"] and manifest["files"][path.name]["hash"] == fhash:
        return {"status": "skipped", "file": path.name, "chunks": manifest["files"][path.name]["chunks"]}

    _p(f"📄 Loading {path.name}…")
    docs = load_pdf(path)

    _p(f"✂️ Chunking {len(docs)} pages…")
    chunks = chunk_documents(docs)

    vs = get_vectorstore()
    # Remove any old chunks for this file (re-ingest case)
    try:
        vs.delete(where={"source": path.name})
    except Exception:
        pass

    _p(f"🧠 Embedding {len(chunks)} chunks (this is the slow step on CPU)…")
    vs.add_documents(chunks)

    manifest["files"][path.name] = {
        "hash": fhash,
        "chunks": len(chunks),
        "pages": len({d.metadata.get("page") for d in docs}),
    }
    save_manifest(manifest)

    _p("🔍 Rebuilding BM25 sparse index…")
    all_docs = vs.get()
    rebuilt = [
        Document(page_content=t, metadata=m)
        for t, m in zip(all_docs["documents"], all_docs["metadatas"])
    ]
    rebuild_bm25(rebuilt)

    return {"status": "ingested", "file": path.name, "chunks": len(chunks)}


def ingest_all_uploads(progress_cb=None) -> list[dict]:
    results = []
    for p in sorted(UPLOAD_DIR.glob("*.pdf")):
        results.append(ingest_file(p, progress_cb=progress_cb))
    return results


def list_indexed_files() -> list[dict]:
    m = load_manifest()
    return [{"name": k, **v} for k, v in m["files"].items()]


def remove_file(name: str) -> None:
    vs = get_vectorstore()
    try:
        vs.delete(where={"source": name})
    except Exception:
        pass
    m = load_manifest()
    m["files"].pop(name, None)
    save_manifest(m)
    # Remove physical file
    p = UPLOAD_DIR / name
    if p.exists():
        p.unlink()
    # Rebuild BM25
    all_docs = vs.get()
    rebuilt = [
        Document(page_content=t, metadata=m_)
        for t, m_ in zip(all_docs["documents"], all_docs["metadatas"])
    ]
    rebuild_bm25(rebuilt)


def clear_index() -> None:
    vs = get_vectorstore()
    try:
        vs.delete_collection()
    except Exception:
        pass
    if BM25_PICKLE.exists():
        BM25_PICKLE.unlink()
    if MANIFEST_PATH.exists():
        MANIFEST_PATH.unlink()
