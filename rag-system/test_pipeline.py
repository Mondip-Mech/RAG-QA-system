"""Diagnostic — runs each pipeline stage directly and shows where it fails."""
import traceback
from backend.config import SETTINGS
from backend.ingestion import get_vectorstore, load_bm25, list_indexed_files

print("=" * 60)
print("STEP 1: Check what's indexed")
print("=" * 60)
files = list_indexed_files()
print(f"Manifest reports {len(files)} files: {files}")

vs = get_vectorstore()
data = vs.get()
print(f"Chroma actually has {len(data['documents'])} chunks")

bm25 = load_bm25()
print(f"BM25 loaded: {bm25 is not None}")

if not data["documents"]:
    print("\n❌ KB IS EMPTY — re-upload your PDF and wait for it to fully ingest")
    raise SystemExit(1)

print("\n" + "=" * 60)
print("STEP 2: Test retrieval")
print("=" * 60)
from backend.retriever import hybrid_retrieve
try:
    docs = hybrid_retrieve("what is this document about")
    print(f"✓ Retrieved {len(docs)} candidates")
    if docs:
        print(f"  First doc preview: {docs[0].page_content[:100]}...")
except Exception:
    print("❌ Retrieval failed:")
    traceback.print_exc()
    raise SystemExit(1)

print("\n" + "=" * 60)
print("STEP 3: Test reranker")
print("=" * 60)
from backend.reranker import rerank
try:
    ranked = rerank("what is this document about", docs)
    print(f"✓ Reranker returned {len(ranked)} docs")
except Exception:
    print("❌ Reranker failed:")
    traceback.print_exc()
    raise SystemExit(1)

print("\n" + "=" * 60)
print("STEP 4: Test generation (this is where most issues happen)")
print("=" * 60)
from backend.generator import stream_answer
try:
    print("Streaming answer:")
    full = ""
    for token, _ in stream_answer("what is this document about", ranked, history=[]):
        print(token, end="", flush=True)
        full += token
    print(f"\n\n✓ Got {len(full)} chars total")
except Exception:
    print("\n❌ Generation failed:")
    traceback.print_exc()
    raise SystemExit(1)

print("\n" + "=" * 60)
print("✅ ALL STAGES WORKED — the issue is in Streamlit, not the pipeline")
print("=" * 60)