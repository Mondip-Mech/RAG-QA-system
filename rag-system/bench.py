"""Bench each pipeline stage. Will hang on whichever step is broken."""
import time
import sys

print("Starting bench...", flush=True)

print("Importing modules...", flush=True)
from backend.retriever import hybrid_retrieve, rewrite_query
from backend.reranker import rerank
from backend.generator import stream_answer
print("Imports done.", flush=True)

q = "what is this document about"

print("\n--- Hybrid retrieve ---", flush=True)
t0 = time.time()
cands = hybrid_retrieve(q)
print(f"  {time.time()-t0:.1f}s -- {len(cands)} candidates", flush=True)

print("\n--- Rerank ---", flush=True)
t0 = time.time()
ranked = rerank(q, cands)
print(f"  {time.time()-t0:.1f}s -- {len(ranked)} docs", flush=True)

print("\n--- Generate (streaming) ---", flush=True)
t0 = time.time()
full = ""
for token, _ in stream_answer(q, ranked, history=[]):
    sys.stdout.write(token)
    sys.stdout.flush()
    full += token
print(f"\n  {time.time()-t0:.1f}s -- {len(full)} chars", flush=True)

print("\nDONE.", flush=True)