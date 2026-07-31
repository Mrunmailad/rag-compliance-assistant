"""
test_retrieval.py

Sanity check: search the FAISS index with a real question and see what
comes back, before wiring retrieval into the full Gemini-powered pipeline.

Usage:
    python scripts/test_retrieval.py "your question here"
"""

import json
import sys
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

INDEX_PATH = Path("data/processed/faiss_index.bin")
LOOKUP_PATH = Path("data/processed/chunk_lookup.json")
MODEL_NAME = "all-MiniLM-L6-v2"

TOP_K = 5


def main():
    query = " ".join(sys.argv[1:]) or "How do I register a private limited company?"

    print(f"Query: {query}\n")

    index = faiss.read_index(str(INDEX_PATH))
    with open(LOOKUP_PATH, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    model = SentenceTransformer(MODEL_NAME)
    query_vec = model.encode([query], normalize_embeddings=True).astype("float32")

    scores, indices = index.search(query_vec, TOP_K)

    print(f"Top {TOP_K} results:\n" + "-" * 60)
    for rank, (score, idx) in enumerate(zip(scores[0], indices[0]), start=1):
        chunk = chunks[idx]
        preview = chunk["text"][:200].replace("\n", " ")
        print(
            f"\n[{rank}] score={score:.3f}  "
            f"{chunk['act_short']} — Section {chunk['section_number']}: "
            f"{chunk['section_title']}"
        )
        print(f"    {preview}...")


if __name__ == "__main__":
    main()