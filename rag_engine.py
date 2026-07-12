"""
rag_engine.py
Turns retrieved chunks + chat history into a prompt the LLM can answer from —
and only from. This is the "generation" half of RAG; vector_store.py is the
"retrieval" half. Keeping them in separate modules mirrors the actual
indexing/retrieval/generation separation the pipeline is built around.

Also builds the prompts for three LLM-assisted extras that reuse the same
"answer only from what's given" discipline as the main chat flow:
  - post-upload document summarization
  - suggested-question generation
  - multi-document comparison
"""

from providers import get_completion

RAG_SYSTEM_INSTRUCTIONS = """You are an enterprise knowledge assistant. Answer the user's \
question using ONLY the context excerpts provided below — do not use outside knowledge, \
and do not guess. If the context doesn't contain the answer, say so plainly.

Every claim you make must end with a citation in this exact format: [source: page]. \
Use the source and page values exactly as given in each excerpt. If a source has no \
page number, cite it as [source]. If your answer draws on multiple excerpts, cite each \
one after the sentence it supports.

Context excerpts:
{context}
"""

SUMMARY_SYSTEM_INSTRUCTIONS = """You write short, factual summaries of documents for a \
document-intelligence tool. Summarize ONLY what is in the excerpts below — do not \
speculate about content that isn't shown. Write 3-5 sentences, plain prose, no \
markdown headers, no citations needed."""

SUGGESTED_QUESTIONS_SYSTEM_INSTRUCTIONS = """You generate example questions a user might \
ask about a document, based only on the excerpts below. Output EXACTLY {n} questions, \
one per line, no numbering, no bullets, no extra commentary — just the questions \
themselves. Each question must be answerable from the excerpts shown."""

COMPARISON_SYSTEM_INSTRUCTIONS = """You are an enterprise knowledge assistant comparing \
multiple documents. Use ONLY the excerpts provided below, grouped by document. \
Structure your answer as:
1. A short paragraph of key similarities.
2. A short paragraph of key differences.
3. Anything notably present in one document and absent in the others.
Every claim must end with a citation in this exact format: [source: page]. If a source \
has no page number, cite it as [source]. Do not use outside knowledge, and say so \
plainly if the excerpts don't support a comparison on some point.

Documents to compare:
{context}
"""

MAX_CONTEXT_CHUNKS_FOR_EXTRAS = 6  # keep summarization/suggestions cheap and fast


def format_context(hits: list[dict]) -> str:
    """Turns retrieved chunks into a labeled block the LLM can cite from."""
    if not hits:
        return "(no relevant context was found in the uploaded documents)"

    blocks = []
    for h in hits:
        label = f"{h['source']}, page {h['page']}" if h['page'] else h['source']
        blocks.append(f"--- Excerpt from {label} ---\n{h['text']}")
    return "\n\n".join(blocks)


def build_system_prompt(hits: list[dict], base_system_prompt: str = "") -> str:
    prompt = RAG_SYSTEM_INSTRUCTIONS.format(context=format_context(hits))
    if base_system_prompt.strip():
        prompt += f"\n\nAdditional instructions from the user:\n{base_system_prompt.strip()}"
    return prompt


def format_source_label(hit: dict) -> str:
    return f"{hit['source']} · page {hit['page']}" if hit['page'] else hit['source']


# ---------------------------------------------------------------------------
# Automatic document summarization
# ---------------------------------------------------------------------------

def summarize_document(chunks: list, model: str, api_keys: dict) -> dict:
    """Summarizes a freshly-processed document from its first few chunks
    (in document order, not retrieval order — there's no query yet). Called
    once right after process_file() succeeds.
    chunks: list of document_processor.Chunk
    Returns the same {ok, content, error, tokens, elapsed} shape as get_completion."""
    if not chunks:
        return {"ok": False, "content": "", "error": "No chunks to summarize.", "tokens": None, "elapsed": 0.0}

    sample = sorted(chunks, key=lambda c: c.chunk_index)[:MAX_CONTEXT_CHUNKS_FOR_EXTRAS]
    hits = [{"source": c.source, "page": c.page, "text": c.text} for c in sample]
    system_prompt = SUMMARY_SYSTEM_INSTRUCTIONS + "\n\nExcerpts:\n" + format_context(hits)
    messages = [{"role": "user", "content": "Summarize this document."}]
    return get_completion(model, messages, system_prompt, api_keys)


# ---------------------------------------------------------------------------
# Suggested-question generation
# ---------------------------------------------------------------------------

def generate_suggested_questions(chunks: list, model: str, api_keys: dict, n: int = 4) -> dict:
    """Same first-few-chunks sampling as summarize_document(). Returns
    {ok, content, error, tokens, elapsed}; on success `content` is the raw
    LLM output — call parse_suggested_questions() on it to get a list."""
    if not chunks:
        return {"ok": False, "content": "", "error": "No chunks available.", "tokens": None, "elapsed": 0.0}

    sample = sorted(chunks, key=lambda c: c.chunk_index)[:MAX_CONTEXT_CHUNKS_FOR_EXTRAS]
    hits = [{"source": c.source, "page": c.page, "text": c.text} for c in sample]
    system_prompt = SUGGESTED_QUESTIONS_SYSTEM_INSTRUCTIONS.format(n=n) + "\n\nExcerpts:\n" + format_context(hits)
    messages = [{"role": "user", "content": "Generate the questions now."}]
    return get_completion(model, messages, system_prompt, api_keys)


def parse_suggested_questions(raw: str, n: int = 4) -> list[str]:
    """Cleans up numbering/bullets the model might add despite instructions,
    and caps the result at n questions."""
    lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
    cleaned = []
    for ln in lines:
        ln = ln.lstrip("-*•").strip()
        # Strip leading "1.", "1)", "Q1:" style numbering.
        for sep in [". ", ") ", ": "]:
            if sep in ln[:5]:
                head, _, rest = ln.partition(sep)
                if head.strip("Qq").strip().isdigit() or head.strip().isdigit():
                    ln = rest.strip()
                break
        if ln:
            cleaned.append(ln)
    return cleaned[:n]


# ---------------------------------------------------------------------------
# Document comparison
# ---------------------------------------------------------------------------

def build_comparison_prompt(hits_by_document: dict[str, list[dict]]) -> str:
    """hits_by_document: {filename: [hit, ...]} — one retrieval pass already
    run per selected document, so each document contributes its own
    most-relevant excerpts to the comparison rather than competing in a
    single top-k (which would let one large/verbose document crowd out the
    others)."""
    blocks = []
    for source, hits in hits_by_document.items():
        blocks.append(f"=== {source} ===\n{format_context(hits)}")
    context = "\n\n".join(blocks)
    return COMPARISON_SYSTEM_INSTRUCTIONS.format(context=context)


def compare_documents(hits_by_document: dict[str, list[dict]], question: str,
                       model: str, api_keys: dict) -> dict:
    system_prompt = build_comparison_prompt(hits_by_document)
    messages = [{"role": "user", "content": question}]
    return get_completion(model, messages, system_prompt, api_keys)

