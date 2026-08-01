# Business Registration & Compliance Assistant

A RAG-based (Retrieval-Augmented Generation) assistant that answers questions about company and LLP registration under Indian law, with every answer grounded in and cited against real statutory text — not general AI knowledge.

**Live demo:** https://rag-compliance-assistant-tu8k.onrender.com

---

## What this does

Ask a question like *"How do I register a private limited company?"* and the assistant:
1. Retrieves the most relevant sections from real Indian legal texts using semantic search (FAISS)
2. Passes only that retrieved text to an LLM (Gemini), with strict instructions to answer **only** from the provided context
3. Returns an answer with explicit citations (Act name + Section number) for every claim
4. Explicitly declines to answer questions outside its knowledge base, rather than guessing

## Why RAG instead of just asking an LLM directly

General-purpose LLMs can hallucinate specific legal details (wrong section numbers, outdated provisions, invented requirements) with high confidence. By retrieving real statutory text first and constraining generation to that context, this assistant can show its work — every claim traces back to a specific, verifiable section of law, and the system will say "I don't know" rather than fabricate an answer when the source material doesn't cover something.

## Architecture

```
Data collection -> PDF extraction -> Section-level chunking -> Embeddings (fastembed/ONNX)
-> FAISS vector index -> [User question] -> Semantic retrieval (top-5) -> Gemini
(context-constrained generation) -> Cited answer + retrieved context shown in UI
```

## Data sources

Currently covers:
- **The Companies Act, 2013** (India Code)
- **The Limited Liability Partnership Act, 2008** (India Code)
- **The Companies (Incorporation) Rules, 2014** (IBC Laws — the government portal only served this document in Hindi at time of collection)

**Not yet covered** (by design, communicated transparently in the UI sidebar): GST registration, Udyam/MSME registration. Natural extensions of the same pipeline, planned as future work.

## Tech stack

- **Embeddings:** `fastembed` (ONNX Runtime) — see "Why fastembed" below for why this replaced `sentence-transformers`
- **Vector search:** FAISS (`IndexFlatIP`, cosine similarity via normalized inner product)
- **LLM:** Google Gemini (`gemini-flash-latest`) — free tier, no cost to run
- **PDF parsing:** `pdfplumber`
- **UI:** Streamlit, custom-themed
- **Deployment:** Render (free tier)

## Project structure

```
├── app.py                        # Streamlit UI (main entry point)
├── requirements.txt
├── .streamlit/config.toml        # Theme configuration
├── data/
│   ├── raw/                      # Source PDFs
│   └── processed/                # Extracted chunks, FAISS index
└── scripts/
    ├── extract_and_chunk.py      # PDF -> section-level text chunks
    ├── build_index.py            # Chunks -> embeddings -> FAISS index
    ├── ask.py                    # CLI version of the RAG pipeline
    ├── test_retrieval.py         # Test retrieval in isolation
    └── test_gemini.py            # Verify Gemini API key works
```

## Running locally

```bash
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Create a `.env` file at the project root:
```
GEMINI_API_KEY=your_key_here
```

Get a free key at [Google AI Studio](https://aistudio.google.com/apikey) — no credit card required.

To rebuild the data pipeline from scratch:
```bash
python scripts/extract_and_chunk.py
python scripts/build_index.py
```

Then run the app:
```bash
streamlit run app.py
```

---

## Engineering journey: what actually went wrong, and how it got fixed

This project went through several real debugging cycles rather than working end-to-end on the first try. Documenting them here because the debugging process is itself part of the engineering story.

### 1. Section-boundary detection (extraction)

**Problem:** the first version of the chunking regex produced duplicate section numbers (the same number appearing 3-4 times) and large gaps in the sequence (missing sections 4, 5, 7, 9...).

**Root cause:** cross-references inside a section's body text (e.g., *"...as specified in section 6..."*) were being misdetected as new section headers, since they happened to sit at the start of a wrapped line.

**Fix:** switched from matching complete single-line headers to a two-step approach — detect any line that *could* be a header, then only accept it if its section number is numerically greater than the last accepted one. Bare acts are sequentially numbered by law, so any candidate that doesn't increase the count is virtually always a false positive.

### 2. Forms/Annexures leaking into section content

**Problem:** the Companies (Incorporation) Rules document has specimen forms attached after the actual rules, with their own numbered paragraphs. The monotonic filter above happily accepted these as "continuing rules" since their numbers kept increasing, inflating the rule count from ~35 (correct) to 87 (wrong).

**Fix:** added detection for standalone Form/Annexure/Schedule headings and truncated extraction before that point — but only when the heading was a genuine standalone line, not a mid-sentence reference to a form (e.g. *"...shall be filed in Form No. INC-1..."*).

### 3. The corpus-integrity bug (the big one)

**Problem:** after the pipeline was "working" and the app was deployed, spot-checking retrieval quality revealed something serious: **290 out of 403 chunks (72% of the corpus) contained only a section's title, with no actual legal text.** A previously-correct answer (minimum number of directors) started failing, which is what triggered the investigation.

**Root cause:** Indian bare acts list every section title in a "Table of Contents" at the very front of the PDF, in perfect numeric order (1, 2, 3... 470). The monotonic-number filter from fix #1 — designed to catch cross-reference false positives — happily accepted this entire table of contents as if it were real section content, since it genuinely is in increasing order. This consumed the counter up to 470 immediately. When the *real* section bodies appeared later in the document (also numbered 1 through 470), each one was silently **rejected**, because by then the counter was already at its maximum — no later section number could ever be "greater than the last accepted one" again.

**Fix:** added a step to skip past the table of contents entirely before running section-boundary detection, using the standard enacting-clause boilerplate ("BE it enacted by Parliament...") that reliably marks where the table of contents ends and real numbered content begins in Indian legislation.

**Result:** chunk count went from 403 (72% broken) to 588 (99%+ genuinely substantive), and previously-failing questions started returning correct, properly-cited answers.

### 4. Deployment memory crashes

**Problem:** after deploying to Render's free tier (512MB RAM), the app repeatedly crashed and silently restarted every few minutes — visible in logs as `"Uvicorn server started"` recurring with a new process ID over and over, rather than running continuously.

**Root cause:** `sentence-transformers` depends on PyTorch, which by default installs GPU-oriented CUDA libraries even though Render's free tier has no GPU. This bloated both install size and runtime memory well past the 512MB ceiling.

**First attempt:** switched to the CPU-only PyTorch build (`--extra-index-url https://download.pytorch.org/whl/cpu`). This reduced install size but the app still crashed — the embedding model itself, once loaded, was still too heavy for 512MB alongside everything else.

**Actual fix:** replaced `sentence-transformers` (PyTorch-based) with `fastembed` (ONNX Runtime-based) — same embedding job, meaningfully lighter memory footprint since it doesn't carry PyTorch's runtime overhead at all. This required rebuilding the FAISS index from scratch, since switching embedding backends changes the vector space. After this switch, the crash-restart loop stopped entirely.

### Why this history matters

Each of these was a genuine, verifiable failure mode caught through actual testing rather than assumed away — and each was root-caused rather than patched around. The corpus-integrity bug in particular is a good example of why RAG systems need retrieval-quality verification, not just a working demo: the app was fully functional and "looked done" while 72% of its knowledge base was silently empty.

---

## Known limitations

- **Scope:** only covers company/LLP incorporation — no GST, tax, labor law, or other compliance areas yet.
- **PDF artifacts:** some chunk text still includes minor print-formatting artifacts (page numbers, watermark text) from source PDFs. Doesn't affect correctness, not fully cleaned.
- **Retrieved-context transparency:** the UI shows the top 5 retrieved chunks per answer, which are not always identical to the chunks the LLM actually cited in its final text — normal RAG behavior, but worth understanding when reviewing the "retrieved context" panel.

## Future work

- Add GST and Udyam/MSME registration coverage (same pipeline, additional sources)
- Hybrid retrieval (BM25 + vector search) to improve exact-term matching for section-number lookups
- Formal evaluation set with precision/recall metrics against a curated set of test questions

## Disclaimer

This tool is for informational purposes only and is not a substitute for professional legal advice. Always verify against official government sources before making compliance decisions.