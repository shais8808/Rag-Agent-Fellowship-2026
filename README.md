# 📄 DocIQ — Enterprise Document Intelligence Platform

Upload documents, ask questions in natural language, get answers with source citations.
Built as a RAG (retrieval-augmented generation) pipeline on top of the Orbit chat
workspace's UI shell — same theming and provider abstraction, extended with real
document processing, embeddings, and semantic search.

## Architecture

```
Sign up / log in (auth.py -- PBKDF2-hashed local accounts, users.json)
    -> per-user ChromaDB collection ("documents_<username>")

Upload (PDF/TXT/MD/DOCX)
    -> document_processor.py   extract -> clean -> chunk (with page metadata)
    -> vector_store.py         embed (sentence-transformers) -> store (ChromaDB)
                                -> auto summary + suggested questions (rag_engine.py)

Question
    -> langchain_pipeline.py   ConversationalRetrievalChain + ConversationBufferMemory
         -> vector_store.py    hybrid retrieval: BM25 (lexical) + semantic, fused with RRF
         -> rag_engine.py      citation-instructed QA prompt
         -> providers.py       call the selected LLM (Gemini / DeepSeek / Llama)
    -> app.py                  render answer + Sources panel + token/cost readout

Document comparison
    -> vector_store.py         hybrid retrieval run once per selected document
    -> rag_engine.py           comparison prompt (similarities / differences / unique points)
    -> providers.py            single LLM call over all documents' excerpts
```

| File | Responsibility |
|---|---|
| `app.py` | Streamlit UI — auth gate, upload panel, document library, chat window, comparison, stats |
| `auth.py` | Real local authentication: PBKDF2-SHA256 password hashing, `users.json` store |
| `document_processor.py` | Extract, clean, and chunk PDF/TXT/MD/DOCX files |
| `vector_store.py` | Embeddings (sentence-transformers) + ChromaDB storage + hybrid (BM25+semantic) search |
| `langchain_pipeline.py` | LangChain `ConversationalRetrievalChain` + `ConversationBufferMemory`, custom retriever/LLM wrappers |
| `rag_engine.py` | Citation-aware QA prompt, document summarization, suggested questions, comparison prompts |
| `providers.py` | LLM calls — Gemini, DeepSeek, Llama 3.3 (via GitHub Models) |
| `cost_estimator.py` | Per-model $/token pricing table and blended cost estimates |
| `style.py` | Dark/light theme |

## Features implemented

- **Real authentication** — sign up / log in with PBKDF2-SHA256-hashed passwords in a local `users.json`; each user gets an isolated document collection
- Multi-file upload: PDF, TXT, Markdown, DOCX
- Per-file processing status: pages, chunks, success/error
- Embeddings via `all-MiniLM-L6-v2` (CPU-friendly, no GPU needed), stored in ChromaDB
- **Hybrid search** — BM25 keyword search fused with semantic search via Reciprocal Rank Fusion, so exact terms (IDs, acronyms) and paraphrased questions both retrieve well
- **Automatic document summarization** — a short summary is generated right after a document is processed
- **Suggested questions** — 4 example questions per document, shown as clickable buttons that populate the chat box
- **Document comparison** — pick 2+ documents and get a similarities/differences/unique-points comparison, each grounded in that document's own excerpts
- **Full LangChain pipeline** — `ConversationalRetrievalChain` with `ConversationBufferMemory`, backed by a custom LangChain retriever (hybrid search) and a custom LangChain LLM (wraps the multi-provider call layer)
- **Token cost estimation** — blended $/token pricing per model, shown per message and as a running total in the sidebar
- Chat interface with conversation memory (via LangChain's `ConversationBufferMemory`, not just resent raw history)
- Every answer cited as `[source: page]`, plus a "Sources" panel with the exact excerpts
- Document management: view, delete, refresh embeddings per document
- Metadata filtering — restrict search to specific documents
- Chat export to Markdown
- Token usage + cost dashboard (documents, chunks, tokens, estimated cost)
- Dark mode

**Not implemented (left as extensions):** OCR for scanned PDFs, streaming responses, multi-session
chat history, role-based permissions.

## Run locally

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Copy `.env.example` to `.env` and add your real API keys:
   ```bash
   cp .env.example .env
   ```
3. Run the app:
   ```bash
   streamlit run app.py
   ```

The first run downloads the `all-MiniLM-L6-v2` embedding model (~80MB) from Hugging
Face — needs internet once, then it's cached locally (`~/.cache/huggingface`).
ChromaDB persists to a local `./chroma_db` folder, so your indexed documents survive
an app restart.

## Deploy to Streamlit Community Cloud (free)

1. Push this project to a GitHub repo. `.gitignore` already excludes `.env`,
   `chroma_db/`, and uploaded documents.
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
3. **New app** → select your repo → main file `app.py`.
4. **Advanced settings → Secrets**, paste:
   ```toml
   GEMINI_API_KEY = "your_real_key"
   GITHUB_TOKEN = "your_real_token"
   ```
5. Deploy. First build takes a few minutes — it has to download both the Python
   dependencies and the embedding model.

**Note on persistence:** Streamlit Community Cloud's filesystem is ephemeral — the
`chroma_db` folder, `users.json`, and any uploaded documents reset whenever the app
restarts or redeploys. That's fine for a demo/evaluation. For a real deployment, point
`PERSIST_DIR` in `vector_store.py` and `USERS_FILE` in `auth.py` at a mounted volume,
or swap in a real database/identity provider.

**Note on `users.json`:** it holds PBKDF2 password hashes, not plaintext passwords, but
treat it as a secret file anyway — add it to `.gitignore` alongside `.env` and never
commit it.

## Getting API keys (free tiers)
- **Gemini**: [aistudio.google.com/apikey](https://aistudio.google.com/apikey)
- **GitHub Models** (DeepSeek, Llama): [github.com/settings/tokens](https://github.com/settings/tokens) — classic token with `models: read` scope

## CPU-only / 8GB RAM notes
- Embedding model: `all-MiniLM-L6-v2` — 80MB, fast on CPU, no GPU dependency.
- Chunk size is 800 characters with 120 overlap — a reasonable default; increase
  overlap if you notice answers missing context that spans a chunk boundary.
- ChromaDB was chosen over FAISS because it gives metadata filtering (search
  specific documents) for free — FAISS would need that layer built by hand.
