"""
DocIQ — Enterprise Document Intelligence Platform
Upload documents, ask questions in natural language, get cited answers.
Run with: streamlit run app.py

Built on the Orbit chat workspace's UI shell (session pattern, provider
abstraction, theming) — extended with a real RAG pipeline: document
processing -> hybrid (BM25 + semantic) retrieval -> a LangChain
ConversationalRetrievalChain with memory -> cited generation, plus
real authentication, document summarization, suggested questions,
document comparison, and token-cost estimation.
"""

import os
import time
from datetime import datetime
from typing import Any

# Fix for joblib/loky incompatibility with Streamlit hot reload
os.environ["JOBLIB_START_METHOD"] = "spawn"
os.environ["LOKY_MAX_CPU_COUNT"] = "1"

import re

import streamlit as st
from dotenv import load_dotenv

import auth
from cost_estimator import estimate_cost, format_cost, total_cost
from document_processor import process_file, ProcessResult
from langchain_pipeline import get_memory, run_chain
from providers import MODEL_CATALOG, get_completion
from rag_engine import (
    compare_documents,
    format_source_label,
    generate_suggested_questions,
    parse_suggested_questions,
    summarize_document,
)
from style import get_css
from vector_store import VectorStore

load_dotenv()

PAGE_TITLE = " RAG Document Agent"
SUPPORTED_TYPES = ["pdf", "txt", "md", "markdown", "docx"]
TOP_K = 4
SUGGESTED_QUESTIONS_N = 4

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

def init_state() -> None:
    st.session_state.setdefault("username", "")
    st.session_state.setdefault("auth_mode", "Log in")
    st.session_state.setdefault("dark_mode", False)
    st.session_state.setdefault("messages", [])            # [{role, content, sources, tokens, elapsed, model}]
    st.session_state.setdefault("doc_status", {})           # {filename: ProcessResult}
    st.session_state.setdefault("file_cache", {})           # {filename: raw bytes} — needed for "refresh embeddings"
    st.session_state.setdefault("doc_summaries", {})        # {filename: summary str}
    st.session_state.setdefault("doc_questions", {})        # {filename: [suggested question, ...]}
    st.session_state.setdefault("model", list(MODEL_CATALOG.keys())[0])
    st.session_state.setdefault("system_prompt", "")
    st.session_state.setdefault("selected_sources", [])     # metadata filter: search only these docs
    st.session_state.setdefault("compare_selection", [])
    st.session_state.setdefault("user_input", "")
    st.session_state.setdefault("_clear_input", False)
    st.session_state.setdefault("lc_memory", get_memory())  # ConversationBufferMemory, one per chat session

    if st.session_state.get("_clear_input"):
        st.session_state.user_input = ""
        st.session_state["_clear_input"] = False


def get_api_keys() -> dict[str, str]:
    return {
        "gemini": os.getenv("GEMINI_API_KEY", ""),
        "github": os.getenv("GITHUB_TOKEN", ""),
        "openrouter": os.getenv("OPENROUTER_API_KEY", ""),
    }


@st.cache_resource(show_spinner=False)
def get_vector_store(username: str) -> VectorStore:
    """Cached per-username so each authenticated user gets an isolated
    document collection — st.cache_resource keys its cache on the function's
    arguments, so passing `username` here is what gives every user their own
    VectorStore/Chroma collection instead of one shared global one."""
    slug = re.sub(r"[^a-z0-9_-]", "_", username.lower()).strip("_") or "user"
    collection_name = f"documents_{slug}"[:63]
    return VectorStore(collection_name=collection_name)


# ---------------------------------------------------------------------------
# Real authentication (sign up / log in) — replaces the old name-only gate
# ---------------------------------------------------------------------------

def render_auth_gate() -> bool:
    """Real local authentication (see auth.py): PBKDF2-hashed passwords in a
    local users.json file. Returns True once the user is logged in."""
    if st.session_state.username:
        return True

    st.markdown(
        '<div class="di-hero"><div class="di-hero-mark">📄</div>'
        '<div class="di-hero-title">RAG Document Agent</div>'
        '<div class="di-hero-sub">Sign in to start your session</div></div>',
        unsafe_allow_html=True,
    )

    _, center, _ = st.columns([1, 2, 1])
    with center:
        mode = st.radio("Mode", ["Log in", "Sign up"], horizontal=True, label_visibility="collapsed")

        if mode == "Log in":
            with st.form("login_form"):
                username = st.text_input("Username")
                password = st.text_input("Password", type="password")
                submitted = st.form_submit_button("Log in", use_container_width=True)
            if submitted:
                result = auth.log_in(username, password)
                if result.ok:
                    st.session_state.username = username.strip().lower()
                    st.rerun()
                else:
                    st.error(result.error)
        else:
            with st.form("signup_form"):
                username = st.text_input("Choose a username")
                password = st.text_input("Choose a password", type="password")
                confirm = st.text_input("Confirm password", type="password")
                st.caption("At least 8 characters, mixed case, and a number.")
                submitted = st.form_submit_button("Create account", use_container_width=True)
            if submitted:
                result = auth.sign_up(username, password, confirm)
                if result.ok:
                    st.success("Account created — you can log in now.")
                else:
                    st.error(result.error)

    # st.caption("Passwords are hashed locally (PBKDF2-SHA256) and stored in ./users.json — no external identity provider.")
    return False


# ---------------------------------------------------------------------------
# Sidebar — upload panel
# ---------------------------------------------------------------------------

def _status_icon(status: str) -> str:
    return "✅" if status == "success" else "⚠️"


def _run_post_upload_extras(name: str, result: ProcessResult, model: str, api_keys: dict) -> None:
    """Automatic summarization + suggested-question generation, run once
    right after a document is successfully processed. Best-effort: a missing
    API key or a provider error just means no summary/questions for that
    document, not a failed upload — the document is still indexed either way."""
    summary_result = summarize_document(result.chunks, model, api_keys)
    if summary_result["ok"]:
        st.session_state.doc_summaries[name] = summary_result["content"]

    questions_result = generate_suggested_questions(result.chunks, model, api_keys, n=SUGGESTED_QUESTIONS_N)
    if questions_result["ok"]:
        st.session_state.doc_questions[name] = parse_suggested_questions(
            questions_result["content"], n=SUGGESTED_QUESTIONS_N
        )


def process_uploaded_files(files: list, vs: VectorStore, model: str, api_keys: dict) -> None:
    """Shared processing pipeline used by both the sidebar uploader and the
    big central drag-and-drop dropzone shown before any documents exist."""
    progress = st.progress(0.0, text="Starting…")
    for i, f in enumerate(files):
        progress.progress((i) / len(files), text=f"Processing {f.name}…")
        raw = f.read()
        st.session_state.file_cache[f.name] = raw
        result = process_file(f.name, raw)
        st.session_state.doc_status[f.name] = result
        if result.status == "success":
            vs.add_chunks(result.chunks)
            progress.progress((i + 0.5) / len(files), text=f"Summarizing {f.name}…")
            _run_post_upload_extras(f.name, result, model, api_keys)
    progress.progress(1.0, text="Done")
    time.sleep(0.3)
    progress.empty()
    st.rerun()


def render_upload_panel(vs: VectorStore, model: str, api_keys: dict) -> None:
    st.markdown('<div class="di-date-label">Add documents</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="di-dropzone-compact">'
        '<div class="di-hero-mark di-hero-mark-sm">📤</div>',
        unsafe_allow_html=True,
    )
    uploaded = st.file_uploader(
        "Upload", type=SUPPORTED_TYPES, accept_multiple_files=True, label_visibility="collapsed",
        help="PDF, TXT, Markdown, DOCX", key="sidebar_uploader",
    )
    st.markdown('</div>', unsafe_allow_html=True)

    if uploaded and st.button("⚙️ Process documents", use_container_width=True):
        process_uploaded_files(uploaded, vs, model, api_keys)

    if st.session_state.doc_status:
        with st.expander(f"📋 Processing status ({len(st.session_state.doc_status)})", expanded=False):
            for name, result in st.session_state.doc_status.items():
                icon = _status_icon(result.status)
                if result.status == "success":
                    st.markdown(
                        f"{icon} **{name}** — {result.num_pages} page(s), {result.num_chunks} chunk(s)"
                    )
                    summary = st.session_state.doc_summaries.get(name)
                    if summary:
                        st.caption(summary)
                else:
                    st.markdown(f"{icon} **{name}** — {result.error}")


# ---------------------------------------------------------------------------
# Sidebar — document library
# ---------------------------------------------------------------------------

def render_document_library(vs: VectorStore) -> None:
    st.markdown('<div class="di-date-label">Document library</div>', unsafe_allow_html=True)
    docs = vs.list_documents()

    if not docs:
        st.caption("No documents indexed yet.")
        return

    for name, chunk_count in docs.items():
        with st.container():
            col1, col2, col3 = st.columns([5, 1, 1])
            col1.markdown(f"📄 **{name}**  \n<span class='di-meta'>{chunk_count} chunks</span>",
                           unsafe_allow_html=True)
            if col2.button("🔄", key=f"refresh_{name}", help="Re-process and refresh embeddings"):
                raw = st.session_state.file_cache.get(name)
                if raw is None:
                    st.warning(f"Original file for '{name}' isn't cached in this session — re-upload to refresh.")
                else:
                    result = process_file(name, raw)
                    st.session_state.doc_status[name] = result
                    if result.status == "success":
                        vs.refresh_document(name, result.chunks)
                        st.rerun()
            if col3.button("🗑️", key=f"delete_{name}", help="Delete this document"):
                vs.delete_document(name)
                st.session_state.doc_status.pop(name, None)
                st.session_state.file_cache.pop(name, None)
                st.session_state.doc_summaries.pop(name, None)
                st.session_state.doc_questions.pop(name, None)
                st.session_state.selected_sources = [
                    s for s in st.session_state.selected_sources if s != name
                ]
                st.session_state.compare_selection = [
                    s for s in st.session_state.compare_selection if s != name
                ]
                st.rerun()

            questions = st.session_state.doc_questions.get(name)
            if questions:
                with st.expander("💡 Suggested questions", expanded=False):
                    for q in questions:
                        if st.button(q, key=f"suggest_{name}_{q[:24]}", use_container_width=True):
                            st.session_state.user_input = q
                            st.rerun()

    st.session_state.selected_sources = st.multiselect(
        "Search only within", options=list(docs.keys()),
        default=[s for s in st.session_state.selected_sources if s in docs],
        help="Leave empty to search across all documents (metadata filtering).",
    )


# ---------------------------------------------------------------------------
# Sidebar — document comparison
# ---------------------------------------------------------------------------

def render_document_comparison(vs: VectorStore, api_keys: dict) -> None:
    docs = vs.list_documents()
    if len(docs) < 2:
        return

    st.markdown('<div class="di-date-label">Compare documents</div>', unsafe_allow_html=True)
    st.session_state.compare_selection = st.multiselect(
        "Documents to compare", options=list(docs.keys()),
        default=[s for s in st.session_state.compare_selection if s in docs],
        key="compare_multiselect",
        help="Pick 2 or more documents to compare side by side.",
    )
    focus = st.text_input(
        "Comparison focus (optional)", placeholder="e.g. pricing terms, key findings…",
        key="compare_focus",
    )

    if st.button("⚖️ Compare selected", use_container_width=True,
                 disabled=len(st.session_state.compare_selection) < 2):
        run_document_comparison(vs, api_keys, st.session_state.compare_selection, focus)


def run_document_comparison(vs: VectorStore, api_keys: dict, selection: list[str], focus: str) -> None:
    """Runs one retrieval pass per selected document (rather than a single
    top-k across all of them) so every document contributes its own most
    relevant excerpts instead of one verbose document crowding out the rest."""
    question = focus.strip() or "Compare these documents."
    hits_by_document: dict[str, list[dict]] = {}
    with st.spinner("Gathering excerpts from each document…"):
        for name in selection:
            hits_by_document[name] = vs.hybrid_query(question, top_k=TOP_K, source_filter=[name])

    with st.spinner("Comparing documents…"):
        result = compare_documents(hits_by_document, question, st.session_state.model, api_keys)

    flattened_sources = [h for hits in hits_by_document.values() for h in hits]
    if result["ok"]:
        st.session_state.messages.append({
            "role": "user", "content": f"📊 Compare: {', '.join(selection)}" + (f" — focus: {focus}" if focus.strip() else ""),
        })
        st.session_state.messages.append({
            "role": "assistant", "content": result["content"], "sources": flattened_sources,
            "tokens": result["tokens"], "elapsed": result["elapsed"], "model": st.session_state.model,
        })
    else:
        st.session_state.messages.append({
            "role": "assistant", "content": f"⚠️ **Comparison failed:** {result['error']}",
            "sources": [], "tokens": None, "elapsed": result.get("elapsed"), "model": st.session_state.model,
        })
    st.rerun()


# ---------------------------------------------------------------------------
# Sidebar — model, stats, actions
# ---------------------------------------------------------------------------

def render_model_and_settings() -> None:
    st.markdown('<div class="di-date-label">Model</div>', unsafe_allow_html=True)
    models = list(MODEL_CATALOG.keys())
    st.session_state.model = st.selectbox(
        "Model", models, index=models.index(st.session_state.model), label_visibility="collapsed",
    )
    with st.expander("⚙️ Extra instructions", expanded=False):
        st.session_state.system_prompt = st.text_area(
            "Extra instructions", value=st.session_state.system_prompt,
            label_visibility="collapsed", height=90,
        )


def build_transcript() -> str:
    lines = [f"# DocIQ chat — {st.session_state.username}\n"]
    for m in st.session_state.messages:
        who = "**You**" if m["role"] == "user" else f"**DocIQ ({m.get('model', '')})**"
        lines.append(f"{who}:\n\n{m['content']}\n")
        if m.get("sources"):
            src_lines = "\n".join(f"- {format_source_label(s)}" for s in m["sources"])
            lines.append(f"*Sources:*\n{src_lines}\n")
    return "\n---\n\n".join(lines)


def render_stats_and_actions(vs: VectorStore) -> None:
    st.markdown('<div class="di-date-label">Statistics</div>', unsafe_allow_html=True)
    total_tokens = sum(m.get("tokens") or 0 for m in st.session_state.messages if m["role"] == "assistant")
    est_cost = total_cost(st.session_state.messages)
    col1, col2 = st.columns(2)
    col1.metric("Documents", len(vs.list_documents()))
    col2.metric("Chunks stored", vs.total_chunks())
    col1.metric("Tokens used", total_tokens)
    col2.metric("Est. cost", format_cost(est_cost))
    st.caption("Cost is a blended estimate from list pricing, not a real bill — see cost_estimator.py.")

    st.session_state.dark_mode = st.toggle("🌙 Dark mode", value=st.session_state.dark_mode)

    if st.session_state.messages:
        st.download_button(
            "⬇️ Export chat", data=build_transcript(),
            file_name=f"dociq_chat_{datetime.now():%Y%m%d_%H%M}.md",
            mime="text/markdown", use_container_width=True,
        )
    if st.button("🗑️ Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.lc_memory = get_memory()
        st.rerun()
    if st.button("🚪 Log out", use_container_width=True):
        st.session_state.username = ""
        st.session_state.messages = []
        st.session_state.lc_memory = get_memory()
        st.rerun()


def render_sidebar(vs: VectorStore, api_keys: dict) -> None:
    with st.sidebar:
        st.markdown(
            f'<div class="di-brand"><span class="di-brand-mark">📄</span> DocIQ</div>'
            f'<div class="di-signed-in">Signed in as {auth.get_display_name(st.session_state.username)}</div>',
            unsafe_allow_html=True,
        )
        render_upload_panel(vs, st.session_state.model, api_keys)
        st.divider()
        render_document_library(vs)
        st.divider()
        render_document_comparison(vs, api_keys)
        st.divider()
        render_model_and_settings()
        st.divider()
        render_stats_and_actions(vs)


# ---------------------------------------------------------------------------
# Main chat area
# ---------------------------------------------------------------------------

def render_upload_hero(vs: VectorStore, model: str, api_keys: dict) -> None:
    """Big centered drag-and-drop dropzone shown before any document has
    been indexed — the centerpiece empty state. Real native browser
    drag-and-drop: files can be dropped straight onto this card."""
    st.markdown(
        """
        <div class="di-hero">
            <div class="di-hero-mark">📤</div>
            <div class="di-hero-title">Please upload your documents</div>
            <div class="di-hero-sub">Drag and drop files below, or browse — then ask DocIQ anything about them.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    uploaded = st.file_uploader(
        "Upload", type=SUPPORTED_TYPES, accept_multiple_files=True, label_visibility="collapsed",
        help="PDF, TXT, Markdown, DOCX", key="hero_uploader",
    )
    if uploaded:
        _, center, _ = st.columns([1, 1, 1])
        with center:
            with st.container(key="di_hero_process_btn"):
                if st.button("⚙️ Process documents", use_container_width=True, key="hero_process_btn"):
                    process_uploaded_files(uploaded, vs, model, api_keys)


def render_ready_hero() -> None:
    """Shown once documents are indexed but no question has been asked yet."""
    st.markdown(
        """
        <div class="di-hero">
            <div class="di-hero-mark">📄</div>
            <div class="di-hero-title">Ask your documents anything</div>
            <div class="di-hero-sub">Your documents are indexed and ready — ask a question below.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sources(sources: list[dict]) -> None:
    if not sources:
        return
    with st.expander(f"📎 Sources ({len(sources)})"):
        for s in sources:
            score = s.get("score")
            relevance = f"{score}" if score is not None else "n/a (keyword match)"
            st.markdown(
                f"**{format_source_label(s)}** · relevance {relevance}\n\n"
                f"<span class='di-meta'>{s['text'][:280]}{'…' if len(s['text']) > 280 else ''}</span>",
                unsafe_allow_html=True,
            )
            st.markdown("---")


def render_messages(vs: VectorStore, model: str, api_keys: dict) -> None:
    if not st.session_state.messages:
        if vs.total_chunks() == 0:
            render_upload_hero(vs, model, api_keys)
        else:
            render_ready_hero()
        return

    for m in st.session_state.messages:
        avatar = "🧑" if m["role"] == "user" else "📄"
        with st.chat_message(m["role"], avatar=avatar):
            st.markdown(m["content"])
            if m["role"] == "assistant":
                bits = []
                if m.get("elapsed") is not None:
                    bits.append(f"⏱ {m['elapsed']:.2f}s")
                if m.get("tokens") is not None:
                    bits.append(f"🔢 {m['tokens']} tokens")
                    cost = estimate_cost(m.get("model", ""), m["tokens"]).usd
                    bits.append(f"💰 {format_cost(cost)}")
                if bits:
                    st.markdown(f'<div class="di-meta">{" · ".join(bits)}</div>', unsafe_allow_html=True)
                render_sources(m.get("sources", []))


def render_composer() -> bool:
    st.write("")
    input_col, send_col = st.columns([9, 1])
    with input_col:
        st.text_area(
            "Message", key="user_input", placeholder="Ask a question about your documents…",
            height=68, label_visibility="collapsed",
        )
    with send_col:
        st.write("")
        with st.container(key="di_send_btn"):
            return st.button("➤", use_container_width=True)


def handle_send(vs: VectorStore, api_keys: dict[str, str]) -> None:
    question = st.session_state.user_input.strip()
    if not question:
        st.warning("Please enter a question first.")
        return

    if vs.total_chunks() == 0:
        st.warning("Upload and process at least one document before asking questions.")
        return

    st.session_state.messages.append({"role": "user", "content": question})

    source_filter = st.session_state.selected_sources or None

    with st.spinner("Agent is retrieving and generating an answer…"):
        # Full LangChain pipeline: ConversationalRetrievalChain (hybrid
        # BM25+semantic retriever) + ConversationBufferMemory, so follow-up
        # questions are resolved with real conversational memory rather than
        # just resending the raw message list.
        result = run_chain(
            vs, st.session_state.model, api_keys, st.session_state.lc_memory,
            question, top_k=TOP_K, source_filter=source_filter,
        )

    if result["ok"]:
        st.session_state.messages.append({
            "role": "assistant", "content": result["content"], "sources": result.get("sources", []),
            "tokens": result["tokens"], "elapsed": result["elapsed"], "model": st.session_state.model,
        })
    else:
        st.session_state.messages.append({
            "role": "assistant", "content": f"⚠️ **Error:** {result['error']}",
            "sources": [], "tokens": None, "elapsed": result.get("elapsed"), "model": st.session_state.model,
        })

    st.session_state["_clear_input"] = True
    st.rerun()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    st.set_page_config(page_title=PAGE_TITLE, page_icon="📄", layout="wide", initial_sidebar_state="expanded")
    init_state()
    st.markdown(get_css(st.session_state.dark_mode), unsafe_allow_html=True)

    if not render_auth_gate():
        return

    vs = get_vector_store(st.session_state.username)
    api_keys = get_api_keys()

    render_sidebar(vs, api_keys)
    render_messages(vs, st.session_state.model, api_keys)

    if render_composer():
        handle_send(vs, api_keys)


if __name__ == "__main__":
    main()
