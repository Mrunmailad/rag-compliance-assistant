"""
build_index.py

Generates sentence embeddings for every chunk in data/processed/chunks.json
and builds a FAISS index for fast similarity search.

Usage:
    python scripts/build_index.py

Reads from:  data/processed/chunks.json
Writes to:   data/processed/faiss_index.bin
             data/processed/chunk_lookup.json   (id -> chunk metadata, same order as index)
"""

import json
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

CHUNKS_PATH = Path("data/processed/chunks.json")
INDEX_PATH = Path("data/processed/faiss_index.bin")
LOOKUP_PATH = Path("data/processed/chunk_lookup.json")

MODEL_NAME = "all-MiniLM-L6-v2"  # small, fast, runs fully on CPU


def build_embedding_text(chunk: dict) -> str:
    """
    What we actually embed: prepend the act name + section title to the
    body text. This helps retrieval match questions phrased in plain
    English (e.g. "how do I register a company") against formal legal
    section titles (e.g. "Incorporation of company").
    """
    return (
        f"{chunk['act_name']}, Section {chunk['section_number']}: "
        f"{chunk['section_title']}\n\n{chunk['text']}"
    )


def main():
    if not CHUNKS_PATH.exists():
        print(f"{CHUNKS_PATH} not found. Run scripts/extract_and_chunk.py first.")
        return

    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    print(f"Loaded {len(chunks)} chunks.")
    print(f"Loading embedding model ({MODEL_NAME}) ...")
    model = SentenceTransformer(MODEL_NAME)

    texts = [build_embedding_text(c) for c in chunks]

    print("Generating embeddings (this may take a minute on CPU) ...")
    embeddings = model.encode(
        texts,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,  # so we can use cosine similarity via inner product
    )
    embeddings = embeddings.astype("float32")

    dimension = embeddings.shape[1]
    print(f"Embedding dimension: {dimension}")

    # IndexFlatIP = exact search via inner product; since vectors are
    # normalized, inner product == cosine similarity. Simple and fast
    # enough for a corpus this size (no need for approximate search).
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)

    faiss.write_index(index, str(INDEX_PATH))
    print(f"Wrote FAISS index to {INDEX_PATH.resolve()}")

    # Save a parallel lookup so we can map a FAISS result row -> chunk metadata
    with open(LOOKUP_PATH, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)
    print(f"Wrote chunk lookup to {LOOKUP_PATH.resolve()}")

    print(f"\nDone. Index contains {index.ntotal} vectors.")


if __name__ == "__main__":
    main()
    