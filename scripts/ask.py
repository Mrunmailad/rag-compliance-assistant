"""
ask.py

The full RAG pipeline: retrieve relevant law sections for a question,
then ask Gemini to answer using ONLY that retrieved context, citing
the specific act + section for each claim.

Usage:
    python scripts/ask.py "your question here"
"""

import json
import os
import sys
from pathlib import Path

import faiss
from dotenv import load_dotenv
from google import genai
from sentence_transformers import SentenceTransformer

load_dotenv()

INDEX_PATH = Path("data/processed/faiss_index.bin")
LOOKUP_PATH = Path("data/processed/chunk_lookup.json")
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
LLM_MODEL_NAME = "gemini-flash-latest"
TOP_K = 5

SYSTEM_PROMPT = """You are a compliance assistant that answers questions about \
Indian business registration law (Companies Act 2013, LLP Act 2008, and the \
Companies Incorporation Rules 2014).

Rules you MUST follow:
1. Answer ONLY using the provided context below. Do not use outside knowledge.
2. If the context does not contain enough information to answer, say so clearly \
instead of guessing.
3. For every claim, cite the specific source in this format: \
(Source: [Act Short Name], Section [number]).
4. Keep the answer clear and practical, as if explaining to a business owner \
with no legal background.
5. Do not invent section numbers or details not present in the context.
"""


def retrieve(query: str, embed_model: SentenceTransformer, index, chunks: list[dict]):
    query_vec = embed_model.encode([query], normalize_embeddings=True).astype("float32")
    scores, indices = index.search(query_vec, TOP_K)
    return [
        {**chunks[idx], "score": float(score)}
        for score, idx in zip(scores[0], indices[0])
    ]


def build_context_block(results: list[dict]) -> str:
    blocks = []
    for r in results:
        blocks.append(
            f"[{r['act_short']}, Section {r['section_number']}: {r['section_title']}]\n"
            f"{r['text']}"
        )
    return "\n\n---\n\n".join(blocks)


def main():
    query = " ".join(sys.argv[1:])
    if not query:
        print('Usage: python scripts/ask.py "your question here"')
        return

    print(f"Question: {query}\n")
    print("Retrieving relevant sections ...")

    embed_model = SentenceTransformer(EMBED_MODEL_NAME)
    index = faiss.read_index(str(INDEX_PATH))
    with open(LOOKUP_PATH, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    results = retrieve(query, embed_model, index, chunks)
    context = build_context_block(results)

    print("Asking Gemini ...\n")

    api_key = os.getenv("GEMINI_API_KEY")
    client = genai.Client(api_key=api_key)

    prompt = f"{SYSTEM_PROMPT}\n\nCONTEXT:\n{context}\n\nQUESTION: {query}\n\nANSWER:"

    response = client.models.generate_content(
        model=LLM_MODEL_NAME,
        contents=prompt,
    )

    print("=" * 70)
    print("ANSWER")
    print("=" * 70)
    print(response.text)

    print("\n" + "=" * 70)
    print("SOURCES RETRIEVED")
    print("=" * 70)
    for r in results:
        print(f"  - {r['act_short']}, Section {r['section_number']} "
              f"(similarity: {r['score']:.3f})")


if __name__ == "__main__":
    main()