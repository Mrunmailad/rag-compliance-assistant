"""
extract_and_chunk.py

Extracts text from bare-act PDFs (Companies Act 2013, LLP Act 2008,
Companies Incorporation Rules 2014) and splits them into section-level
chunks suitable for embedding + retrieval.

Usage:
    python scripts/extract_and_chunk.py

Reads from:  data/raw/*.pdf
Writes to:   data/processed/chunks.json
"""

import json
import re
from pathlib import Path

import pdfplumber

# ---- Configuration: map filenames to source metadata ----
SOURCES = {
    "companies_act_2013.pdf": {
        "act_name": "The Companies Act, 2013",
        "act_short": "Companies Act 2013",
        "doc_type": "act",
        "max_section": 470,
    },
    "llp_act_2008.pdf": {
        "act_name": "The Limited Liability Partnership Act, 2008",
        "act_short": "LLP Act 2008",
        "doc_type": "act",
        "max_section": 81,
    },
    "companies_incorporation_rules_2014.pdf": {
        "act_name": "The Companies (Incorporation) Rules, 2014",
        "act_short": "Incorporation Rules 2014",
        "doc_type": "rules",
        "max_section": 45,
    },
}

RAW_DIR = Path("data/raw")
OUT_DIR = Path("data/processed")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Matches the START of a section header line, e.g.:
#   "7. Incorporation of company."
#   "3A. Members severally liable in certain cases."
#   "378A. Application of Act to..."
# We only need to detect the START reliably — the title text itself is
# grabbed separately below, since it may wrap onto a second line in the PDF.
SECTION_START_PATTERN = re.compile(r"^\s*(\d{1,3}[A-Z]{0,2})\.\s+(.+)")


def _numeric_part(section_number: str) -> int:
    """'378A' -> 378, '10' -> 10 -- used to check numbers are increasing."""
    digits = re.match(r"\d+", section_number)
    return int(digits.group()) if digits else -1


def extract_pdf_text(pdf_path: Path) -> str:
    """Extract raw text from all pages of a PDF, joined with page markers."""
    full_text = []
    with pdfplumber.open(pdf_path) as pdf:
        total_pages = len(pdf.pages)
        for i, page in enumerate(pdf.pages):
            text = page.extract_text() or ""
            full_text.append(text)
            if (i + 1) % 25 == 0 or (i + 1) == total_pages:
                print(f"    ...page {i + 1}/{total_pages}")
    return "\n".join(full_text)



# Matches a line that is ENTIRELY a form/annexure/schedule heading and
# nothing else -- e.g. "FORM No. INC-9" or "ANNEXURE-A" on its own line.
# A mid-sentence reference like "...shall be filed in Form No. INC-1 within
# thirty days" has more words after the form code, so it won't match.
CUTOFF_PATTERN = re.compile(
    r"^\s*(FORM\s+No\.?\s*[A-Za-z0-9\-]{1,10}\.?"
    r"|ANNEXURE[\s\-]?[A-Za-z0-9]{0,3}"
    r"|SCHEDULE[\s\-]?[IVXLC0-9]{0,5})\s*$",
    re.IGNORECASE,
)


# Indian bare acts (not subordinate rules) list every section title in a
# "Arrangement of Sections" table of contents at the very front of the PDF,
# in perfect numeric order. This false-positively satisfies the monotonic
# filter in split_into_sections, consuming the counter before any real
# section body is ever reached -- meaning genuine section text later in
# the document gets silently rejected as "not increasing". The standard
# enacting-clause boilerplate below reliably marks where the real numbered
# content begins, right after the table of contents ends.
ENACTING_CLAUSE_PATTERN = re.compile(r"BE it enacted by Parliament", re.IGNORECASE)


def skip_toc_and_preamble(text: str) -> str:
    """Cut off everything before the Act's enacting clause, if present."""
    match = ENACTING_CLAUSE_PATTERN.search(text)
    if match:
        return text[match.end():]
    return text


def truncate_before_forms(text: str) -> str:
    """Cut off the text at the first standalone Form/Annexure/Schedule heading, if any."""
    lines = text.split("\n")
    for idx, line in enumerate(lines):
        if CUTOFF_PATTERN.match(line.strip()):
            return "\n".join(lines[:idx])
    return text


def split_into_sections(text: str, max_section: int = 10_000) -> list[dict]:
    """
    Split raw act/rules text into a list of {section_number, section_title, text}
    using numbered-section headers as boundaries.

    Strategy: find every line that *could* be a section header (starts with
    "N." or "NA."), then keep only the ones whose numeric part is strictly
    greater than the previously accepted section number AND within the
    known valid range for this document (max_section). Bare acts are
    sequentially numbered by law, so any "match" that doesn't increase the
    count is virtually always a false positive -- usually a cross-reference
    inside another section's body text (e.g. "...as per section 6...").
    The max_section ceiling additionally rejects matches that drift into
    unrelated numbered content further in the document (e.g. clause numbers
    inside inline form references) that would otherwise still look like a
    validly increasing sequence.
    """
    lines = text.split("\n")

    candidates = []  # (line_index, section_number, numeric_value, rest_of_line)
    for idx, line in enumerate(lines):
        m = SECTION_START_PATTERN.match(line.strip())
        if m:
            num = m.group(1)
            candidates.append((idx, num, _numeric_part(num), m.group(2).strip()))

    # Keep only candidates that increase the section count and stay within
    # the known valid range for this document.
    accepted = []
    last_value = -1
    for idx, num, value, rest in candidates:
        if value > max_section:
            continue
        if value >= last_value:
            accepted.append((idx, num, rest))
            last_value = value

    sections = []
    for i, (idx, num, rest) in enumerate(accepted):
        end_idx = accepted[i + 1][0] if i + 1 < len(accepted) else len(lines)
        body = "\n".join(lines[idx:end_idx]).strip()

        # Skip near-empty captures
        if len(body) < 40:
            continue

        # Title = text up to the first period after the number, falling back
        # to the first ~100 chars if no clean sentence boundary is found
        title_match = re.match(r"([^.]{3,120})\.", rest)
        title = title_match.group(1).strip() if title_match else rest[:100].strip()

        sections.append(
            {
                "section_number": num,
                "section_title": title,
                "text": body,
            }
        )
    return sections


def main():
    all_chunks = []
    chunk_id = 0

    pdf_files = list(RAW_DIR.glob("*.pdf"))
    if not pdf_files:
        print(f"No PDFs found in {RAW_DIR.resolve()}. Check your data/raw folder.")
        return

    for pdf_path in pdf_files:
        meta = SOURCES.get(pdf_path.name)
        if not meta:
            print(f"⚠️  Skipping {pdf_path.name} — not in SOURCES config. "
                  f"Add it to the SOURCES dict if you want it processed.")
            continue

        print(f"Processing {pdf_path.name} ...")
        raw_text = extract_pdf_text(pdf_path)
        if meta["doc_type"] == "act":
            raw_text = skip_toc_and_preamble(raw_text)
        raw_text = truncate_before_forms(raw_text)
        sections = split_into_sections(raw_text, max_section=meta["max_section"])
        print(f"  -> found {len(sections)} sections")

        for sec in sections:
            chunk_id += 1
            all_chunks.append(
                {
                    "id": f"chunk_{chunk_id:04d}",
                    "source_file": pdf_path.name,
                    "act_name": meta["act_name"],
                    "act_short": meta["act_short"],
                    "doc_type": meta["doc_type"],
                    "section_number": sec["section_number"],
                    "section_title": sec["section_title"],
                    "text": sec["text"],
                }
            )

    out_path = OUT_DIR / "chunks.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, indent=2, ensure_ascii=False)

    print(f"\nDone. Wrote {len(all_chunks)} chunks to {out_path.resolve()}")


if __name__ == "__main__":
    main()