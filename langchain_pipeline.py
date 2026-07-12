"""
langchain_pipeline.py
Wires the app's existing pieces (VectorStore's hybrid search, providers.py's
multi-provider LLM calls) into an actual LangChain pipeline: a
ConversationBufferMemory-backed ConversationalRetrievalChain, built from a
custom LangChain Retriever and a custom LangChain LLM.

Why custom wrappers instead of e.g. `langchain_chroma.Chroma` + `ChatGoogleGenerativeAI`:
this app already has a hybrid (BM25 + semantic) retriever in vector_store.py
and a provider-agnostic call layer in providers.py that handles three very
differently-shaped APIs (Gemini REST, GitHub Models via an OpenAI-compatible
client) behind one interface. Re-pointing LangChain's official integrations
at those would mean giving up either the hybrid search or the provider
abstraction. Instead, both are wrapped to satisfy LangChain's Retriever and
LLM interfaces, so the app gets a real `ConversationalRetrievalChain` with
real `ConversationBufferMemory` while keeping its own retrieval and
provider logic as the source of truth underneath.
"""

from typing import Any, List, Optional

from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.language_models.llms import LLM
from langchain_core.prompts import PromptTemplate
from langchain_core.retrievers import BaseRetriever
from pydantic import Field

from providers import get_completion
from rag_engine import RAG_SYSTEM_INSTRUCTIONS

QA_PROMPT = PromptTemplate(
    template=RAG_SYSTEM_INSTRUCTIONS + "\n\nQuestion: {question}\nAnswer:",
    input_variables=["context", "question"],
)


class HybridRetriever(BaseRetriever):
    """LangChain BaseRetriever backed by VectorStore.hybrid_query()
    (BM25 + semantic search fused with Reciprocal Rank Fusion)."""

    vector_store: Any
    top_k: int = 4
    source_filter: Optional[List[str]] = None

    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun) -> List[Document]:
        hits = self.vector_store.hybrid_query(query, top_k=self.top_k, source_filter=self.source_filter)
        docs = []
        for h in hits:
            docs.append(Document(
                page_content=h["text"],
                metadata={
                    "source": h["source"],
                    "page": h["page"],
                    "chunk_index": h["chunk_index"],
                    "score": h.get("score"),
                },
            ))
        return docs


class OrbitLLM(LLM):
    """LangChain LLM wrapper around providers.get_completion(), so the same
    Gemini/DeepSeek/Llama call logic (error handling, token accounting) that
    powers the plain chat mode also powers the LangChain chain. LangChain's
    `_call` interface is synchronous and returns a plain string, so
    token/timing metadata that get_completion() normally returns is stashed
    on the instance after each call rather than returned from _call itself.
    """

    model: str
    api_keys: dict

    last_tokens: Optional[int] = Field(default=None, exclude=True)
    last_elapsed: Optional[float] = Field(default=None, exclude=True)
    last_error: Optional[str] = Field(default=None, exclude=True)

    @property
    def _llm_type(self) -> str:
        return "orbit-multi-provider"

    def _call(self, prompt: str, stop: Optional[List[str]] = None, run_manager=None, **kwargs) -> str:
        # The QA prompt already bakes the citation instructions + context
        # into `prompt` itself, so it's sent as a single user turn with no
        # separate system prompt (get_completion still accepts one, but
        # ConversationalRetrievalChain's default prompt shape is single-turn).
        result = get_completion(self.model, [{"role": "user", "content": prompt}], "", self.api_keys)
        self.last_tokens = result.get("tokens")
        self.last_elapsed = result.get("elapsed")
        self.last_error = result.get("error")
        if not result["ok"]:
            return f"⚠️ **Error:** {result['error']}"
        return result["content"]


def get_memory() -> ConversationBufferMemory:
    """Fresh memory object — callers keep exactly one of these per chat
    session (stored in st.session_state), not one per call, so it actually
    accumulates conversation turns."""
    return ConversationBufferMemory(
        memory_key="chat_history",
        input_key="question",
        output_key="answer",
        return_messages=True,
    )


def build_chain(vector_store, model: str, api_keys: dict, memory: ConversationBufferMemory,
                 top_k: int = 4, source_filter: Optional[List[str]] = None) -> tuple[ConversationalRetrievalChain, OrbitLLM]:
    """Builds a ConversationalRetrievalChain over the hybrid retriever, using
    ConversationBufferMemory for history and the citation-instructed QA
    prompt from rag_engine.py. Returns (chain, llm) — the caller reads
    llm.last_tokens / last_elapsed / last_error after invoking the chain,
    since ConversationalRetrievalChain's return dict doesn't include them."""
    llm = OrbitLLM(model=model, api_keys=api_keys)
    retriever = HybridRetriever(vector_store=vector_store, top_k=top_k, source_filter=source_filter)

    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=True,
        combine_docs_chain_kwargs={"prompt": QA_PROMPT},
    )
    return chain, llm


def run_chain(vector_store, model: str, api_keys: dict, memory: ConversationBufferMemory,
              question: str, top_k: int = 4, source_filter: Optional[List[str]] = None) -> dict:
    """One conversational-retrieval turn. Returns the same
    {ok, content, error, tokens, elapsed} shape as providers.get_completion,
    plus 'sources': list[dict] shaped like vector_store hits, so app.py can
    render sources exactly as it already does for the non-LangChain path."""
    chain, llm = build_chain(vector_store, model, api_keys, memory, top_k, source_filter)

    try:
        result = chain.invoke({"question": question})
    except Exception as e:
        return {"ok": False, "content": "", "error": f"LangChain pipeline error: {e}",
                "tokens": None, "elapsed": 0.0, "sources": []}

    if llm.last_error:
        return {"ok": False, "content": "", "error": llm.last_error,
                "tokens": llm.last_tokens, "elapsed": llm.last_elapsed, "sources": []}

    sources = []
    for doc in result.get("source_documents", []):
        sources.append({
            "text": doc.page_content,
            "source": doc.metadata.get("source"),
            "page": doc.metadata.get("page"),
            "chunk_index": doc.metadata.get("chunk_index"),
            "score": doc.metadata.get("score"),
        })

    return {
        "ok": True,
        "content": result.get("answer", ""),
        "error": None,
        "tokens": llm.last_tokens,
        "elapsed": llm.last_elapsed,
        "sources": sources,
    }
