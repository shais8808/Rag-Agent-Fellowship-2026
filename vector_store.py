"""
vector_store.py
Wraps sentence-transformers (embeddings) + ChromaDB (storage/search) behind
a small class so app.py never touches either library directly.

Model choice: all-MiniLM-L6-v2 — 80MB, runs fast on CPU, no GPU required.
This matters because the whole point of this module is to stay usable on a
CPU-only, 8GB-RAM machine.

Also implements hybrid search: BM25 (lexical/keyword) fused with Chroma's
cosine-similarity semantic search via Reciprocal Rank Fusion (RRF). Pure
semantic search misses exact-term matches (IDs, acronyms, product codes,
rare proper nouns) that a keyword search catches; pure BM25 misses
paraphrases. RRF combines both rankings without needing to normalize two
differently-scaled similarity metrics onto the same scale.
"""

import re

import streamlit as st

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
COLLECTION_NAME = "documents"
PERSIST_DIR = "./chroma_db"

RRF_K = 60  # standard smoothing constant from the original RRF paper
_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


@st.cache_resource(show_spinner=False)
def get_embedder():
    """Loaded once per app process and cached — reloading this on every
    rerun would make the UI feel sluggish on a CPU-only machine."""
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(EMBEDDING_MODEL_NAME)


@st.cache_resource(show_spinner=False)
def get_chroma_client():
    import chromadb
    return chromadb.PersistentClient(path=PERSIST_DIR)


class VectorStore:
    """Thin, testable wrapper around one ChromaDB collection."""

    def __init__(self, collection_name: str = COLLECTION_NAME):
        self.embedder = get_embedder()
        self.client = get_chroma_client()
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        # Lazily-built BM25 index over the full corpus. Rebuilt on first read
        # after any write (add/delete/refresh) — see _ensure_bm25().
        self._bm25 = None
        self._bm25_ids: list[str] = []
        self._bm25_docs: list[str] = []
        self._bm25_metas: list[dict] = []
        self._bm25_dirty = True

    # -- writes ------------------------------------------------------------

    def add_chunks(self, chunks: list) -> None:
        """chunks: list of document_processor.Chunk"""
        if not chunks:
            return
        texts = [c.text for c in chunks]
        embeddings = self.embedder.encode(texts, show_progress_bar=False).tolist()
        self.collection.add(
            ids=[c.id for c in chunks],
            embeddings=embeddings,
            documents=texts,
            metadatas=[
                {"source": c.source, "page": c.page if c.page is not None else -1,
                 "chunk_index": c.chunk_index}
                for c in chunks
            ],
        )
        self._bm25_dirty = True

    def delete_document(self, source: str) -> None:
        self.collection.delete(where={"source": source})
        self._bm25_dirty = True

    def refresh_document(self, source: str, chunks: list) -> None:
        """Delete a document's old chunks/embeddings and re-add fresh ones."""
        self.delete_document(source)
        self.add_chunks(chunks)
        self._bm25_dirty = True

    # -- reads ---------------------------------------------------------------

    def query(self, question: str, top_k: int = 4, source_filter: list[str] | None = None) -> list[dict]:
        """Semantic search. Returns retrieved chunks with metadata and distance,
        ordered by relevance (closest first)."""
        if self.collection.count() == 0:
            return []

        query_embedding = self.embedder.encode([question], show_progress_bar=False).tolist()
        where = {"source": {"$in": source_filter}} if source_filter else None

        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=min(top_k, self.collection.count()),
            where=where,
        )

        hits = []
        ids = results.get("ids") or [[]]
        docs = results.get("documents") or [[]]
        metas = results.get("metadatas") or [[]]
        dists = results.get("distances") or [[]]
        for cid, text, meta, dist in zip(ids[0], docs[0], metas[0], dists[0]):
            hits.append({
                "id": cid,
                "text": text,
                "source": meta["source"],
                "page": None if meta["page"] == -1 else meta["page"],
                "chunk_index": meta["chunk_index"],
                "score": round(1 - dist, 3),  # cosine similarity, higher = more relevant
            })
        return hits

    # -- hybrid search (BM25 + semantic, fused with RRF) ---------------------

    def _ensure_bm25(self) -> None:
        """(Re)builds the in-memory BM25 index from the full Chroma corpus.
        Cheap enough to rebuild on demand for the chunk counts this app is
        designed for (CPU-only, 8GB-RAM machine); only runs after a write."""
        if not self._bm25_dirty and self._bm25 is not None:
            return

        from rank_bm25 import BM25Okapi

        if self.collection.count() == 0:
            self._bm25, self._bm25_ids, self._bm25_docs, self._bm25_metas = None, [], [], []
            self._bm25_dirty = False
            return

        got = self.collection.get(include=["documents", "metadatas"])
        self._bm25_ids = got["ids"]
        self._bm25_docs = got["documents"]
        self._bm25_metas = got["metadatas"]
        tokenized = [_tokenize(doc) for doc in self._bm25_docs]
        self._bm25 = BM25Okapi(tokenized)
        self._bm25_dirty = False

    def _bm25_query(self, question: str, candidate_k: int, source_filter: list[str] | None) -> list[str]:
        """Returns chunk ids ranked by BM25 score (best first)."""
        self._ensure_bm25()
        if self._bm25 is None:
            return []

        scores = self._bm25.get_scores(_tokenize(question))
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        ids = []
        for i in ranked:
            if scores[i] <= 0:
                break
            if source_filter and self._bm25_metas[i]["source"] not in source_filter:
                continue
            ids.append(self._bm25_ids[i])
            if len(ids) >= candidate_k:
                break
        return ids

    def hybrid_query(self, question: str, top_k: int = 4,
                      source_filter: list[str] | None = None) -> list[dict]:
        """Hybrid retrieval: BM25 (lexical) ranking fused with semantic
        (cosine) ranking via Reciprocal Rank Fusion. Falls back to pure
        semantic search if BM25 finds no lexical matches at all."""
        if self.collection.count() == 0:
            return []

        candidate_k = max(top_k * 4, 10)
        semantic_hits = self.query(question, top_k=candidate_k, source_filter=source_filter)
        bm25_ids = self._bm25_query(question, candidate_k, source_filter)

        if not bm25_ids:
            return semantic_hits[:top_k]

        by_id = {h["id"]: h for h in semantic_hits}
        semantic_ids = list(by_id.keys())

        rrf_scores: dict[str, float] = {}
        for rank, cid in enumerate(semantic_ids):
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (RRF_K + rank + 1)
        for rank, cid in enumerate(bm25_ids):
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (RRF_K + rank + 1)

        # Any id that came only from BM25 needs its metadata/text pulled in
        # (it wasn't part of the semantic hit list).
        missing = [cid for cid in bm25_ids if cid not in by_id]
        if missing:
            got = self.collection.get(ids=missing, include=["documents", "metadatas"])
            for cid, text, meta in zip(got["ids"], got["documents"], got["metadatas"]):
                by_id[cid] = {
                    "id": cid,
                    "text": text,
                    "source": meta["source"],
                    "page": None if meta["page"] == -1 else meta["page"],
                    "chunk_index": meta["chunk_index"],
                    "score": None,  # no cosine score available for BM25-only hits
                }

        ranked_ids = sorted(rrf_scores, key=lambda cid: rrf_scores[cid], reverse=True)[:top_k]
        results = []
        for cid in ranked_ids:
            hit = dict(by_id[cid])
            hit["rrf_score"] = round(rrf_scores[cid], 4)
            results.append(hit)
        return results

    def list_documents(self) -> dict[str, int]:
        """Returns {filename: chunk_count} for every distinct document currently stored."""
        if self.collection.count() == 0:
            return {}
        all_meta = self.collection.get(include=["metadatas"])["metadatas"]
        counts: dict[str, int] = {}
        for meta in all_meta:
            counts[meta["source"]] = counts.get(meta["source"], 0) + 1
        return counts

    def total_chunks(self) -> int:
        return self.collection.count()
