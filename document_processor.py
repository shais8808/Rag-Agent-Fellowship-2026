"""
document_processor.py
Reads uploaded documents, extracts text, cleans it, and splits it into
citation-ready chunks. Every chunk carries {source, page, chunk_index} so
the RAG layer can always point back to exactly where an answer came from.

Supported formats: PDF, TXT, Markdown, DOCX (bonus).
"""

import re
import uuid
from dataclasses import dataclass, field
from io import BytesIO
from typing import Optional

from pypdf import PdfReader

CHUNK_SIZE = 800
CHUNK_OVERLAP = 120

# Lazy-loaded to avoid joblib/loky initialization during Streamlit reload
_splitter = None

def _get_splitter():
    """Lazy-initialize the text splitter on first use."""
    global _splitter
    if _splitter is None:
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        _splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
    return _splitter


@dataclass
class Chunk:
    id: str
    text: str
    source: str
    page: Optional[int]   # 1-indexed page number, None if the format has no pages
    chunk_index: int      # position within the document (0-indexed, across all pages)


@dataclass
class ProcessResult:
    filename: str
    num_pages: int
    num_chunks: int
    chunks: list[Chunk] = field(default_factory=list)
    status: str = "success"
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Text cleaning
# ---------------------------------------------------------------------------

def clean_text(text: str) -> str:
    """Normalize whitespace and strip control characters without mangling content."""
    text = text.replace("\x00", "")
    # Collapse runs of spaces/tabs, but keep paragraph breaks.
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Drop common PDF extraction artifacts: lone page-number lines, form-feed chars.
    text = text.replace("\x0c", "\n")
    lines = [ln for ln in text.split("\n") if not re.fullmatch(r"\s*\d{1,4}\s*", ln)]
    return "\n".join(lines).strip()


# ---------------------------------------------------------------------------
# Per-format extraction -> list of (page_number_or_None, raw_text)
# ---------------------------------------------------------------------------

def _extract_pdf(file_bytes: bytes) -> list[tuple[int, str]]:
    reader = PdfReader(BytesIO(file_bytes))
    pages = []
    for i, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if text.strip():
            pages.append((i, text))
    return pages


def _extract_docx(file_bytes: bytes) -> list[tuple[Optional[int], str]]:
    import docx  # python-docx
    document = docx.Document(BytesIO(file_bytes))
    full_text = "\n".join(p.text for p in document.paragraphs if p.text.strip())
    return [(None, full_text)] if full_text.strip() else []


def _extract_plain(file_bytes: bytes) -> list[tuple[Optional[int], str]]:
    text = file_bytes.decode("utf-8", errors="ignore")
    return [(None, text)] if text.strip() else []


EXTRACTORS = {
    "pdf": _extract_pdf,
    "docx": _extract_docx,
    "txt": _extract_plain,
    "md": _extract_plain,
    "markdown": _extract_plain,
}


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def process_file(filename: str, file_bytes: bytes) -> ProcessResult:
    """Extract -> clean -> chunk a single uploaded file. Never raises; failures
    are reported in ProcessResult.status/error so the UI can show a per-file
    status row instead of crashing the whole upload batch."""
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    extractor = EXTRACTORS.get(ext)

    if extractor is None:
        return ProcessResult(filename, 0, 0, status="error",
                              error=f"Unsupported file type: .{ext}")

    try:
        raw_pages = extractor(file_bytes)
    except Exception as e:
        return ProcessResult(filename, 0, 0, status="error",
                              error=f"Could not read file: {e}")

    if not raw_pages:
        return ProcessResult(filename, 0, 0, status="error",
                              error="No extractable text found (file may be scanned/image-only).")

    chunks: list[Chunk] = []
    for page_num, raw_text in raw_pages:
        cleaned = clean_text(raw_text)
        if not cleaned:
            continue
        pieces = _get_splitter().split_text(cleaned)
        for piece in pieces:
            chunks.append(Chunk(
                id=str(uuid.uuid4()),
                text=piece,
                source=filename,
                page=page_num,
                chunk_index=len(chunks),
            ))

    num_pages = len(raw_pages) if ext == "pdf" else 1
    return ProcessResult(
        filename=filename,
        num_pages=num_pages,
        num_chunks=len(chunks),
        chunks=chunks,
        status="success" if chunks else "error",
        error=None if chunks else "No chunks produced after cleaning (document may be empty).",
    )
