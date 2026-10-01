"""
Local RAG Knowledge Assistant — Streamlit UI

Run with:
    streamlit run app.py

This app shows the full educational RAG pipeline:

    Documents → Load → Split → Embed → FAISS → Retriever → Prompt → OpenAI → Answer + Sources
"""

from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from src.embeddings import get_embedding_model_name
from src.loaders import SUPPORTED_EXTENSIONS, count_source_files
from src.rag_chain import ask_question, get_llm_display_name
from src.vectorstore import (
    build_vectorstore,
    clear_vectorstore,
    get_chunk_count,
    rebuild_vectorstore,
    vectorstore_exists,
)

load_dotenv()

DOCUMENTS_DIR = Path("data/documents")
VECTORSTORE_DIR = Path("data/vectorstore")


def ensure_folders() -> None:
    """
    Create the documents and vectorstore folders if they do not exist yet.

    Returns:
        None

    Example:
        ensure_folders()
    """
    DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)
    VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)


def save_uploaded_files(uploaded_files) -> list[str]:
    """
    Save uploaded PDF/DOCX files into the local documents folder.

    Args:
        uploaded_files: Streamlit UploadedFile objects from the file uploader.

    Returns:
        list[str]: Names of files that were saved successfully.

    Raises:
        ValueError: If an unsupported file type is uploaded.

    Example:
        saved = save_uploaded_files(uploaded_files)
        print(saved)
    """
    saved_names: list[str] = []

    for uploaded in uploaded_files:
        suffix = Path(uploaded.name).suffix.lower()
        if suffix not in SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported file type: {uploaded.name}. "
                "Only PDF and DOCX files are supported."
            )

        destination = DOCUMENTS_DIR / uploaded.name
        destination.write_bytes(uploaded.getbuffer())
        saved_names.append(uploaded.name)

    return saved_names


def init_session_state() -> None:
    """
    Initialize Streamlit session values used by the chat UI.

    Returns:
        None

    Example:
        init_session_state()
    """
    if "messages" not in st.session_state:
        # Chat history lives only in this Streamlit session (not a database).
        st.session_state.messages = []
    if "chunk_count" not in st.session_state:
        st.session_state.chunk_count = get_chunk_count()


def render_sidebar() -> None:
    """
    Render the knowledge-base controls and status panel.

    Returns:
        None
    """
    with st.sidebar:
        st.markdown(
            """
            <div class="sidebar-brand">
                <div class="brand-icon">◈</div>
                <div>
                    <div class="brand-title">Knowledge Base</div>
                    <div class="brand-subtitle">Manage your document index</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("### Add documents")
        st.caption("Upload PDF or DOCX files to your local knowledge base.")

        uploaded_files = st.file_uploader(
            "Upload PDF / DOCX",
            type=["pdf", "docx"],
            accept_multiple_files=True,
            help="Files are stored under data/documents/",
        )

        if uploaded_files:
            if st.button("Save Uploaded Files", use_container_width=True):
                try:
                    saved = save_uploaded_files(uploaded_files)
                    st.success(f"Saved {len(saved)} file(s).")
                except ValueError as exc:
                    st.error(str(exc))

        col1, col2 = st.columns(2)
        with col1:
            build_clicked = st.button(
                "Build Knowledge Base",
                use_container_width=True,
                type="primary",
            )
        with col2:
            rebuild_clicked = st.button(
                "Rebuild Knowledge Base",
                use_container_width=True,
            )

        if build_clicked:
            with st.spinner("Building knowledge base with HuggingFace embeddings..."):
                try:
                    _, num_docs, num_chunks = build_vectorstore()
                    st.session_state.chunk_count = num_chunks
                    st.success(
                        f"Knowledge base ready — {num_docs} document(s), {num_chunks} chunk(s)."
                    )
                except ValueError as exc:
                    st.error(str(exc))
                except RuntimeError as exc:
                    st.error(str(exc))
                except Exception as exc:
                    st.error(f"Unexpected error while building the knowledge base: {exc}")

        if rebuild_clicked:
            with st.spinner("Rebuilding knowledge base from scratch..."):
                try:
                    _, num_docs, num_chunks = rebuild_vectorstore()
                    st.session_state.chunk_count = num_chunks
                    st.success(
                        f"Knowledge base rebuilt — {num_docs} document(s), {num_chunks} chunk(s)."
                    )
                except ValueError as exc:
                    st.error(str(exc))
                except RuntimeError as exc:
                    st.error(str(exc))
                except Exception as exc:
                    st.error(f"Unexpected error while rebuilding: {exc}")

        if st.button("Clear Vector Store", use_container_width=True):
            clear_vectorstore()
            st.session_state.chunk_count = 0
            st.info("Vector store cleared. Build it again before chatting.")

        st.divider()
        st.markdown("### System status")
        doc_count = count_source_files(str(DOCUMENTS_DIR))
        chunk_count = st.session_state.chunk_count
        store_status = "Ready" if vectorstore_exists() else "Not built"
        status_class = "status-ready" if store_status == "Ready" else "status-warn"

        st.markdown(
            f"""
            <div class="status-card">
                <div class="status-row">
                    <span>Documents</span>
                    <strong>{doc_count}</strong>
                </div>
                <div class="status-row">
                    <span>Chunks</span>
                    <strong>{chunk_count}</strong>
                </div>
                <div class="status-row">
                    <span>Vector Store</span>
                    <span class="status-pill {status_class}">
                        ● {store_status}
                    </span>
                </div>
                <div class="model-block">
                    <span>Embedding model</span>
                    <code>{get_embedding_model_name()}</code>
                </div>
                <div class="model-block">
                    <span>Language model</span>
                    <code>{get_llm_display_name()}</code>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.divider()
        if st.button("Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()


def render_sources(sources, num_chunks: int) -> None:
    """
    Display source citations under an assistant answer.

    Args:
        sources: List of SourceCitation objects.
        num_chunks (int): Number of chunks retrieved for this answer.

    Returns:
        None
    """
    st.markdown(
        f'<div class="retrieval-label">Retrieved <b>{num_chunks}</b> relevant chunk(s)</div>',
        unsafe_allow_html=True,
    )
    if not sources:
        st.markdown("_No sources were returned for this answer._")
        return

    with st.expander(f"📚 Sources  ·  {len(sources)} reference(s)", expanded=False):
        for source in sources:
            if source.page is not None:
                st.markdown(
                    f'<div class="source-item"><span class="source-icon">📄</span>'
                    f'<span><b>{source.filename}</b><br><small>Page {source.page}</small></span></div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f'<div class="source-item"><span class="source-icon">📄</span>'
                    f'<span><b>{source.filename}</b></span></div>',
                    unsafe_allow_html=True,
                )


def main() -> None:
    """
    Launch the Streamlit Local RAG Knowledge Assistant.

    Returns:
        None

    Example:
        # From the project root:
        # streamlit run app.py
        main()
    """
    st.set_page_config(
        page_title="Local RAG Knowledge Assistant",
        page_icon="📚",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.markdown(
        """
        <style>
        /* ---------- Global ---------- */
        .stApp {
            background: #0b0f14;
        }

        .main .block-container {
            max-width: 1180px;
            padding: 2.2rem 2.5rem 6rem 2.5rem;
        }

        [data-testid="stSidebar"] {
            background: #10151c;
            border-right: 1px solid #202833;
        }

        [data-testid="stSidebar"] > div:first-child {
            padding: 1.5rem 1.15rem;
        }

        /* ---------- Typography ---------- */
        h1, h2, h3, p, label, span, div {
            font-family: Inter, ui-sans-serif, system-ui, -apple-system,
                         BlinkMacSystemFont, "Segoe UI", sans-serif;
        }

        h1 {
            letter-spacing: -0.035em;
        }

        /* ---------- Header ---------- */
        .app-header {
            display: flex;
            align-items: center;
            gap: 16px;
            margin: 0 0 2.2rem 0;
        }

        .app-logo {
            width: 52px;
            height: 52px;
            border-radius: 15px;
            display: flex;
            align-items: center;
            justify-content: center;
            background: linear-gradient(135deg, #6d5dfc, #8b7cff);
            box-shadow: 0 10px 35px rgba(109, 93, 252, 0.25);
            font-size: 25px;
        }

        .app-title {
            font-size: 2rem;
            font-weight: 750;
            line-height: 1.1;
            color: #f5f7fb;
        }

        .app-subtitle {
            color: #8f9aaa;
            font-size: 0.94rem;
            margin-top: 5px;
        }

        .pipeline {
            display: flex;
            align-items: center;
            gap: 7px;
            flex-wrap: wrap;
            margin-top: 10px;
            color: #768296;
            font-size: 0.74rem;
        }

        .pipeline-step {
            padding: 5px 9px;
            border: 1px solid #252d38;
            border-radius: 999px;
            background: #121821;
        }

        .pipeline-arrow {
            color: #515c6c;
        }

        /* ---------- Sidebar ---------- */
        .sidebar-brand {
            display: flex;
            align-items: center;
            gap: 11px;
            margin-bottom: 22px;
        }

        .brand-icon {
            width: 39px;
            height: 39px;
            border-radius: 11px;
            display: flex;
            align-items: center;
            justify-content: center;
            background: #6d5dfc;
            color: white;
            font-size: 21px;
        }

        .brand-title {
            color: #f3f5f8;
            font-weight: 700;
            font-size: 1rem;
        }

        .brand-subtitle {
            color: #778294;
            font-size: 0.72rem;
            margin-top: 2px;
        }

        .status-card {
            background: #0c1117;
            border: 1px solid #222b36;
            border-radius: 13px;
            padding: 13px;
            margin-top: 8px;
        }

        .status-row {
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 8px 0;
            color: #8994a4;
            font-size: 0.82rem;
            border-bottom: 1px solid #1c242e;
        }

        .status-row:last-of-type {
            border-bottom: none;
        }

        .status-row strong {
            color: #eef1f5;
        }

        .status-pill {
            font-size: 0.72rem;
            font-weight: 650;
            padding: 3px 8px;
            border-radius: 999px;
        }

        .status-ready {
            color: #73d7a1;
            background: rgba(61, 190, 123, 0.10);
        }

        .status-warn {
            color: #eabf70;
            background: rgba(234, 191, 112, 0.10);
        }

        .model-block {
            margin-top: 11px;
            display: flex;
            flex-direction: column;
            gap: 5px;
        }

        .model-block span {
            color: #687587;
            font-size: 0.69rem;
            text-transform: uppercase;
            letter-spacing: 0.07em;
        }

        .model-block code {
            color: #aeb8c7;
            background: #141a22;
            border: 1px solid #232c37;
            border-radius: 7px;
            padding: 6px 8px;
            font-size: 0.68rem;
            overflow-wrap: anywhere;
        }

        /* ---------- Buttons ---------- */
        .stButton > button {
            border-radius: 9px;
            min-height: 42px;
            border: 1px solid #293340;
            background: #151b23;
            color: #dce2ea;
            font-weight: 600;
            transition: all 0.15s ease;
        }

        .stButton > button:hover {
            border-color: #6659e9;
            color: #ffffff;
            background: #1a202b;
        }

        .stButton > button[kind="primary"] {
            background: #6d5dfc;
            border-color: #6d5dfc;
            color: white;
        }

        .stButton > button[kind="primary"]:hover {
            background: #7a6cff;
            border-color: #7a6cff;
        }

        /* ---------- Chat ---------- */
        [data-testid="stChatMessage"] {
            border: 1px solid #202832;
            border-radius: 15px;
            padding: 1rem 1.1rem;
            margin-bottom: 0.8rem;
            background: #10161e;
        }

        [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] {
            line-height: 1.65;
        }

        [data-testid="stChatInput"] {
            border-color: #2a3440;
        }

        [data-testid="stChatInput"] textarea {
            font-size: 0.95rem;
        }

        /* ---------- Retrieval / Sources ---------- */
        .retrieval-label {
            margin-top: 12px;
            color: #778395;
            font-size: 0.72rem;
        }

        .source-item {
            display: flex;
            align-items: center;
            gap: 10px;
            padding: 9px 0;
            border-bottom: 1px solid #202832;
            color: #dce2ea;
        }

        .source-item:last-child {
            border-bottom: none;
        }

        .source-icon {
            width: 31px;
            height: 31px;
            border-radius: 8px;
            background: #171e28;
            display: flex;
            align-items: center;
            justify-content: center;
        }

        .source-item small {
            color: #7c8798;
        }

        /* ---------- Alerts ---------- */
        [data-testid="stAlert"] {
            border-radius: 11px;
        }

        /* ---------- File uploader ---------- */
        [data-testid="stFileUploader"] {
            border: 1px dashed #303b49;
            border-radius: 11px;
            padding: 4px;
            background: #0d131a;
        }

        /* ---------- Expander ---------- */
        [data-testid="stExpander"] {
            border: 1px solid #242e39;
            border-radius: 11px;
            background: #0e141b;
        }

        /* ---------- Divider ---------- */
        hr {
            border-color: #202832;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    ensure_folders()
    init_session_state()

    st.markdown(
        """
        <div class="app-header">
            <div class="app-logo">📚</div>
            <div>
                <div class="app-title">Local RAG Knowledge Assistant</div>
                <div class="app-subtitle">
                    Ask questions and get grounded answers from your own documents.
                </div>
                <div class="pipeline">
                    <span class="pipeline-step">Documents</span>
                    <span class="pipeline-arrow">→</span>
                    <span class="pipeline-step">FAISS</span>
                    <span class="pipeline-arrow">→</span>
                    <span class="pipeline-step">Retriever</span>
                    <span class="pipeline-arrow">→</span>
                    <span class="pipeline-step">LLM</span>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    render_sidebar()

    if not vectorstore_exists():
        st.info(
            "No knowledge base yet. Upload PDF/DOCX files in the sidebar "
            "(or use the sample documents), then click **Build Knowledge Base**."
        )

    # Show the conversation so far.
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant" and message.get("sources") is not None:
                render_sources(message["sources"], message.get("num_chunks", 0))

    # Chat input for the next user question.
    question = st.chat_input("Ask a question about your documents...")

    if question:
        # Add the user message to session history first.
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Retrieving context and generating an answer..."):
                try:
                    # Pass prior turns only (exclude the just-added user message)
                    # so memory stays separate from the current question.
                    prior_history = st.session_state.messages[:-1]
                    response = ask_question(question, chat_history=prior_history)

                    st.markdown(response.answer)
                    render_sources(response.sources, response.num_chunks)

                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": response.answer,
                            "sources": response.sources,
                            "num_chunks": response.num_chunks,
                        }
                    )
                except FileNotFoundError as exc:
                    st.error(str(exc))
                    st.session_state.messages.pop()
                except ConnectionError as exc:
                    st.error(str(exc))
                    st.session_state.messages.pop()
                except ValueError as exc:
                    st.error(str(exc))
                    st.session_state.messages.pop()
                except RuntimeError as exc:
                    st.error(str(exc))
                    st.session_state.messages.pop()
                except Exception as exc:
                    st.error(f"Something went wrong: {exc}")
                    st.session_state.messages.pop()


if __name__ == "__main__":
    main()
    
