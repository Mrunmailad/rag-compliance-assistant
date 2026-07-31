"""
app.py

Streamlit UI for the RAG-based business registration & compliance assistant.

Usage:
    streamlit run app.py
"""

import json
import os
from pathlib import Path

import faiss
import numpy as np
import streamlit as st
from dotenv import load_dotenv
from fastembed import TextEmbedding
from google import genai

load_dotenv()

INDEX_PATH = Path("data/processed/faiss_index.bin")
LOOKUP_PATH = Path("data/processed/chunk_lookup.json")
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
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
6. Do NOT use markdown headers (no #, ##, or ###). Use bold text and numbered \
lists for structure instead.
"""


def normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1e-9
    return vectors / norms


# ---- Cached resources: loaded once, reused across every user interaction ----

@st.cache_resource
def load_embed_model():
    return TextEmbedding(model_name=EMBED_MODEL_NAME)


@st.cache_resource
def load_index_and_chunks():
    index = faiss.read_index(str(INDEX_PATH))
    with open(LOOKUP_PATH, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    return index, chunks


@st.cache_resource
def load_gemini_client():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        st.error(
            "GEMINI_API_KEY not found. Make sure your .env file is set up "
            "(locally) or the GEMINI_API_KEY secret is configured (on Render)."
        )
        st.stop()
    return genai.Client(api_key=api_key)


def retrieve(query, embed_model, index, chunks):
    query_vec = np.array(list(embed_model.embed([query])), dtype="float32")
    query_vec = normalize(query_vec)
    scores, indices = index.search(query_vec, TOP_K)
    return [
        {**chunks[idx], "score": float(score)}
        for score, idx in zip(scores[0], indices[0])
    ]


def build_context_block(results):
    blocks = []
    for r in results:
        blocks.append(
            f"[{r['act_short']}, Section {r['section_number']}: {r['section_title']}]\n"
            f"{r['text']}"
        )
    return "\n\n---\n\n".join(blocks)


def answer_question(query, embed_model, index, chunks, client):
    results = retrieve(query, embed_model, index, chunks)
    context = build_context_block(results)
    prompt = f"{SYSTEM_PROMPT}\n\nCONTEXT:\n{context}\n\nQUESTION: {query}\n\nANSWER:"

    response = client.models.generate_content(
        model=LLM_MODEL_NAME,
        contents=prompt,
    )
    return response.text, results


def render_sources(results):
    """Render retrieved sources inside a single collapsed expander."""
    with st.expander("⚖️ Retrieved context (top 5 matches)"):
        badges_html = '<div class="cite-row">'
        for r in results:
            badges_html += (
                f'<div class="cite-badge">'
                f'<span class="cite-act">{r["act_short"]}</span>'
                f'<span class="cite-sec">§ {r["section_number"]}</span>'
                f'<span class="cite-score">{r["score"]:.2f}</span>'
                f"</div>"
            )
        badges_html += "</div>"
        st.markdown(badges_html, unsafe_allow_html=True)

        st.markdown('<div style="height: 0.6rem"></div>', unsafe_allow_html=True)

        for r in results:
            st.markdown(
                f'<div class="cite-detail"><span class="cite-detail-ref">'
                f'{r["act_short"]} § {r["section_number"]}</span> — {r["section_title"]}</div>',
                unsafe_allow_html=True,
            )


# ---- Page setup ----

st.set_page_config(
    page_title="Business Registration & Compliance Assistant",
    page_icon="📜",
    layout="centered",
)

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Source+Serif+4:wght@600;700&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@500&display=swap');

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
}

.register-header {
    border-bottom: 2px solid #1F2A44;
    padding-bottom: 1rem;
    margin-bottom: 0.5rem;
}
.register-title {
    font-family: 'Source Serif 4', serif;
    font-weight: 700;
    font-size: clamp(1.25rem, 3.6vw, 1.85rem);
    color: #1F2A44;
    letter-spacing: -0.01em;
    margin-bottom: 0.15rem;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}
.register-eyebrow {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem;
    letter-spacing: 0.12em;
    color: #9C7A3C;
    text-transform: uppercase;
    margin-bottom: 0.3rem;
}
.register-subtitle {
    color: #445070;
    font-size: 0.95rem;
    line-height: 1.5;
}

.disclaimer-strip {
    background: #E8E2D1;
    border-left: 3px solid #9C7A3C;
    padding: 0.7rem 1rem;
    font-size: 0.85rem;
    color: #1F2A44;
    margin: 1rem 0 1.5rem 0;
    border-radius: 2px;
}

.cite-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.68rem;
    letter-spacing: 0.1em;
    color: #9C7A3C;
    margin-top: 0.8rem;
    margin-bottom: 0.4rem;
}
.cite-row {
    display: flex;
    flex-wrap: wrap;
    gap: 0.4rem;
    margin-bottom: 0.3rem;
}
.cite-badge {
    display: flex;
    align-items: center;
    gap: 0.4rem;
    background: #F1EDE2;
    border: 1px solid #C9C2AE;
    border-radius: 4px;
    padding: 0.25rem 0.6rem;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem;
}
.cite-act {
    color: #1F2A44;
    font-weight: 500;
}
.cite-sec {
    color: #9C7A3C;
    font-weight: 500;
}
.cite-score {
    color: #7A7460;
    border-left: 1px solid #C9C2AE;
    padding-left: 0.4rem;
}
.cite-detail {
    font-size: 0.85rem;
    color: #445070;
    padding: 0.15rem 0;
}
.cite-detail-ref {
    font-family: 'IBM Plex Mono', monospace;
    color: #1F2A44;
    font-weight: 500;
}

[data-testid="stSidebar"] {
    background: #E8E2D1;
    border-right: 1px solid #C9C2AE;
}
.sidebar-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 0.1em;
    color: #9C7A3C;
    text-transform: uppercase;
    margin-top: 1rem;
    margin-bottom: 0.5rem;
}
.act-pill {
    display: inline-block;
    background-color: #DCD4C0;
    color: #1F2A44;
    padding: 4px 8px;
    border-radius: 4px;
    font-size: 0.82rem;
    font-weight: 600;
    margin-bottom: 6px;
    border: 1px solid #C9C2AE;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

with st.sidebar:
    st.markdown('<div class="sidebar-label">CORPUS COVERAGE</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="act-pill">Companies Act, 2013</div><br>
        <div class="act-pill">LLP Act, 2008</div><br>
        <div class="act-pill">Companies (Incorporation) Rules, 2014</div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="sidebar-label">NOT YET COVERED</div>', unsafe_allow_html=True)
    st.markdown("• GST registration")
    st.markdown("• Udyam / MSME registration")
    st.caption(
        "This assistant only answers from the acts listed above. "
        "Questions outside this scope will be declined rather than guessed."
    )

st.markdown(
    """
    <div class="register-header">
        <div class="register-eyebrow">RAG-BASED LEGAL RETRIEVAL</div>
        <div class="register-title">📜 Business Registration &amp; Compliance Assistant</div>
        <div class="register-subtitle">
            Ask questions about company or LLP registration under Indian law.
            Every answer is retrieved from and cited against real statutory text.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="disclaimer-strip">
        ⚠️ Informational purposes only — not a substitute for professional legal advice.
        Always verify against official government sources before making compliance decisions.
    </div>
    """,
    unsafe_allow_html=True,
)

embed_model = load_embed_model()
index, chunks = load_index_and_chunks()
client = load_gemini_client()

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    avatar = "🧑‍💼" if msg["role"] == "user" else "💬"
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and "sources" in msg:
            render_sources(msg["sources"])

user_query = st.chat_input("e.g. How do I register a private limited company?")

if user_query:
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user", avatar="🧑‍💼"):
        st.markdown(user_query)

    with st.chat_message("assistant", avatar="💬"):
        with st.spinner("Searching relevant sections and generating answer..."):
            answer, sources = answer_question(user_query, embed_model, index, chunks, client)
        st.markdown(answer)
        render_sources(sources)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": sources}
    )