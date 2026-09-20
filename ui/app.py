import streamlit as st
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import SCRAPE_SOURCES, CHROMA_PERSIST_DIR
from rag.pipeline import RAGPipeline
from scraper.devops_docs import DevOpsDocScraper
from embeddings.local_embeddings import LocalEmbeddings
from vectorstore.chromadb_store import ChromaVectorStore
from config import EMBEDDING_MODEL, CHROMA_COLLECTION

st.set_page_config(page_title="DevOps RAG", page_icon="", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=Inter:wght@400;500;600;700&display=swap');

:root {
    --bg-primary: #0a0e17;
    --bg-secondary: #111827;
    --bg-tertiary: #1a2332;
    --accent: #00d4ff;
    --accent-dim: #0098b3;
    --text-primary: #e2e8f0;
    --text-secondary: #94a3b8;
    --border: #1e293b;
    --user-bubble: #00d4ff;
    --user-text: #0a0e17;
    --assistant-bubble: #1a2332;
    --success: #22c55e;
    --warning: #f59e0b;
    --error: #ef4444;
}

.stApp {
    background: var(--bg-primary) !important;
    color: var(--text-primary) !important;
}

header[data-testid="stHeader"] {
    background: var(--bg-primary) !important;
}

section[data-testid="stSidebar"] {
    background: var(--bg-secondary) !important;
    border-right: 1px solid var(--border) !important;
}

section[data-testid="stSidebar"] .stMarkdown p,
section[data-testid="stSidebar"] .stMarkdown h1,
section[data-testid="stSidebar"] .stMarkdown h2,
section[data-testid="stSidebar"] .stMarkdown h3,
section[data-testid="stSidebar"] label {
    color: var(--text-primary) !important;
}

div[data-testid="stChatMessage"] {
    background: transparent !important;
    border: none !important;
    padding: 0.5rem 0 !important;
}

div[data-testid="stChatMessage"][aria-label="user"] {
    background: transparent !important;
}

div[data-testid="stChatMessage"][aria-label="assistant"] {
    background: transparent !important;
}

div[data-testid="stChatInput"] {
    background: var(--bg-secondary) !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
}

div[data-testid="stChatInput"] textarea {
    color: var(--text-primary) !important;
    font-family: 'Inter', sans-serif !important;
}

.stButton > button {
    background: linear-gradient(135deg, #00d4ff 0%, #0098b3 100%) !important;
    color: var(--user-text) !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    padding: 0.5rem 1.5rem !important;
    transition: all 0.3s ease !important;
    width: 100% !important;
}

.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 20px rgba(0, 212, 255, 0.3) !important;
}

.stSelectbox, .stTextInput, .stMultiSelect {
    font-family: 'Inter', sans-serif !important;
}

code, .stCode {
    font-family: 'JetBrains Mono', monospace !important;
    background: var(--bg-tertiary) !important;
    border-radius: 4px !important;
    padding: 2px 6px !important;
}

pre {
    background: var(--bg-tertiary) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    padding: 1rem !important;
}

.stSpinner > div {
    border-top-color: var(--accent) !important;
}

.welcome-container {
    background: linear-gradient(135deg, rgba(0,212,255,0.1) 0%, rgba(0,152,179,0.05) 100%);
    border: 1px solid rgba(0,212,255,0.2);
    border-radius: 16px;
    padding: 3rem 2rem;
    text-align: center;
    margin: 2rem 0;
}

.welcome-title {
    font-size: 3rem;
    font-weight: 700;
    background: linear-gradient(135deg, #00d4ff 0%, #00ff88 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.5rem;
}

.welcome-subtitle {
    font-size: 1.2rem;
    color: var(--text-secondary);
    margin-bottom: 2rem;
}

.prompt-card {
    background: var(--bg-tertiary);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 1rem 1.2rem;
    margin: 0.5rem;
    cursor: pointer;
    transition: all 0.2s ease;
    text-align: left;
    display: inline-block;
    width: 100%;
    color: var(--text-primary);
    text-decoration: none;
}

.prompt-card:hover {
    border-color: var(--accent);
    background: rgba(0,212,255,0.05);
    transform: translateY(-2px);
}

.prompt-icon {
    font-size: 1.3rem;
    margin-right: 0.5rem;
}

.prompt-title {
    font-weight: 600;
    font-size: 0.95rem;
    margin-bottom: 0.2rem;
}

.prompt-desc {
    font-size: 0.8rem;
    color: var(--text-secondary);
}

.status-badge {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 600;
    margin: 2px;
}

.status-active {
    background: rgba(34, 197, 94, 0.15);
    color: #22c55e;
    border: 1px solid rgba(34, 197, 94, 0.3);
}

.status-inactive {
    background: rgba(148, 163, 184, 0.15);
    color: #94a3b8;
    border: 1px solid rgba(148, 163, 184, 0.3);
}

.source-pill {
    display: inline-block;
    background: rgba(0, 212, 255, 0.1);
    color: #00d4ff;
    border: 1px solid rgba(0, 212, 255, 0.3);
    border-radius: 20px;
    padding: 3px 10px;
    font-size: 0.75rem;
    font-weight: 500;
    margin: 2px;
}

.divider {
    height: 1px;
    background: var(--border);
    margin: 1rem 0;
}

.sidebar-header {
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    color: var(--text-secondary);
    margin-bottom: 0.5rem;
    font-weight: 600;
}

.metric-card {
    background: var(--bg-tertiary);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 1rem;
    text-align: center;
}

.metric-value {
    font-size: 1.8rem;
    font-weight: 700;
    color: var(--accent);
}

.metric-label {
    font-size: 0.8rem;
    color: var(--text-secondary);
}

.stChatMessage {
    font-family: 'Inter', sans-serif !important;
}

.clear-btn {
    background: rgba(239, 68, 68, 0.1) !important;
    color: #ef4444 !important;
    border: 1px solid rgba(239, 68, 68, 0.3) !important;
    border-radius: 8px !important;
    font-weight: 500 !important;
    padding: 0.4rem 1rem !important;
    font-size: 0.85rem !important;
}

.clear-btn:hover {
    background: rgba(239, 68, 68, 0.2) !important;
}

.footer {
    text-align: center;
    padding: 1rem;
    color: var(--text-secondary);
    font-size: 0.8rem;
    border-top: 1px solid var(--border);
    margin-top: 2rem;
}
</style>
""", unsafe_allow_html=True)

if "messages" not in st.session_state:
    st.session_state.messages = []
if "pipeline" not in st.session_state:
    st.session_state.pipeline = None
if "pipeline_status" not in st.session_state:
    st.session_state.pipeline_status = "Not initialized"
if "active_model" not in st.session_state:
    st.session_state.active_model = ""
if "response_time" not in st.session_state:
    st.session_state.response_time = 0

SUGGESTED_PROMPTS = [
    {"icon": "", "title": "Kubernetes", "desc": "Deployments, pods, services, kubectl", "prompt": "How do I deploy an application to Kubernetes?"},
    {"icon": "", "title": "Docker", "desc": "Dockerfiles, Compose, networking", "prompt": "Write a Dockerfile for a Node.js application"},
    {"icon": "", "title": "CI/CD", "desc": "GitHub Actions, pipelines", "prompt": "How to set up a CI/CD pipeline with GitHub Actions?"},
    {"icon": "", "title": "Terraform", "desc": "IaC, providers, state", "prompt": "Explain Terraform state management best practices"},
    {"icon": "", "title": "Monitoring", "desc": "Prometheus, Grafana, alerts", "prompt": "How to configure Prometheus alerting rules?"},
    {"icon": "", "title": "AWS", "desc": "EKS, ECS, EC2, S3", "prompt": "What are the best practices for AWS EKS?"},
]

with st.sidebar:
    st.markdown('<div class="sidebar-header">Configuration</div>', unsafe_allow_html=True)

    provider = st.selectbox(
        "LLM Provider",
        options=["openrouter", "openai", "google"],
        format_func=lambda x: {"openrouter": "OpenRouter", "openai": "OpenAI", "google": "Google AI Studio"}[x],
        label_visibility="collapsed",
    )

    if provider == "openrouter":
        api_key = st.text_input("API Key", type="password", placeholder="sk-or-v1-...", help="Get from https://openrouter.ai")
        default_model = "liquid/lfm-2.5-2.6b:free"
    elif provider == "openai":
        api_key = st.text_input("API Key", type="password", placeholder="sk-...", help="Get from https://platform.openai.com")
        default_model = "gpt-3.5-turbo"
    else:
        api_key = st.text_input("API Key", type="password", placeholder="AIza...", help="Get from https://aistudio.google.com")
        default_model = "gemini-2.0-flash"

    model = st.text_input("Model", value=default_model, label_visibility="collapsed")

    if st.button("Initialize Pipeline", type="primary"):
        if not api_key:
            st.error("Enter an API key first.")
        else:
            with st.spinner("Loading..."):
                st.session_state.pipeline = RAGPipeline(api_key=api_key, model=model, provider=provider)
                st.session_state.pipeline_status = "Active"
                st.session_state.active_model = model
                st.rerun()

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    if st.session_state.pipeline:
        status_class = "status-active"
        status_text = "Active"
    else:
        status_class = "status-inactive"
        status_text = "Not connected"

    st.markdown(f"""
    <div class="metric-card">
        <div style="margin-bottom: 8px;">
            <span class="status-badge {status_class}">{status_text}</span>
        </div>
        <div style="font-size: 0.85rem; color: var(--text-primary); font-weight: 600;">
            {st.session_state.active_model or 'No model'}
        </div>
        <div style="font-size: 0.75rem; color: var(--text-secondary); margin-top: 4px;">
            {provider.title()} Provider
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    selected_sources = st.multiselect(
        "Sources",
        options=list(SCRAPE_SOURCES.keys()),
        default=list(SCRAPE_SOURCES.keys()),
        label_visibility="collapsed",
    )

    if st.button("Ingest Documents"):
        sources_to_scrape = {k: v for k, v in SCRAPE_SOURCES.items() if k in selected_sources}
        progress_text = st.empty()
        progress_bar = st.progress(0)

        scraper = DevOpsDocScraper()
        all_docs = scraper.scrape_all(sources_to_scrape)

        if all_docs:
            progress_text.text("Loading embeddings...")
            embedder = LocalEmbeddings(EMBEDDING_MODEL)
            progress_bar.progress(50)
            texts = [doc.content for doc in all_docs]
            metadatas = [doc.metadata for doc in all_docs]
            progress_text.text("Generating embeddings...")
            embeddings = embedder.embed_documents(texts)
            progress_bar.progress(75)
            progress_text.text("Storing...")
            store = ChromaVectorStore(persist_dir=CHROMA_PERSIST_DIR, collection_name=CHROMA_COLLECTION)
            store.add_documents(texts, embeddings, metadatas)
            progress_bar.progress(100)
            st.success(f"Ingested {len(all_docs)} docs")
            st.rerun()

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    if st.session_state.messages:
        if st.button("Clear Chat", type="secondary"):
            st.session_state.messages = []
            st.rerun()

    st.markdown("""
    <div style="text-align: center; padding: 1rem 0; color: var(--text-secondary); font-size: 0.75rem;">
        DevOps RAG Assistant<br>
        Built with Python + Streamlit
    </div>
    """, unsafe_allow_html=True)

if not st.session_state.messages:
    st.markdown("""
    <div class="welcome-container">
        <div class="welcome-title">DevOps RAG Assistant</div>
        <div class="welcome-subtitle">
            Ask anything about Kubernetes, Docker, CI/CD, Terraform, and more.<br>
            Powered by retrieval-augmented generation from official documentation.
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sidebar-header" style="text-align: center; margin-top: 2rem;">Try asking</div>', unsafe_allow_html=True)

    cols = st.columns(3)
    for i, prompt in enumerate(SUGGESTED_PROMPTS):
        col = cols[i % 3]
        with col:
            if st.button(
                f"{prompt['icon']} {prompt['title']}\n{prompt['desc']}",
                key=f"prompt_{i}",
                use_container_width=True,
            ):
                st.session_state.messages.append({"role": "user", "content": prompt["prompt"]})
                st.rerun()
else:
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.chat_message("user").markdown(msg["content"])
        else:
            with st.chat_message("assistant"):
                st.markdown(msg["content"])
                if "sources" in msg and msg["sources"]:
                    source_html = "".join([f'<span class="source-pill">{s}</span>' for s in msg["sources"]])
                    st.markdown(f'<div style="margin-top: 8px;">{source_html}</div>', unsafe_allow_html=True)
                if "time" in msg:
                    st.markdown(f'<div style="font-size: 0.75rem; color: var(--text-secondary); margin-top: 4px;">Responded in {msg["time"]:.1f}s</div>', unsafe_allow_html=True)

if prompt := st.chat_input("Ask a DevOps question..."):
    if not st.session_state.pipeline:
        st.error("Initialize the pipeline first from the sidebar.")
    else:
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").markdown(prompt)

        with st.chat_message("assistant"):
            status_placeholder = st.empty()
            status_placeholder.markdown("""
            <div style="display: flex; align-items: center; gap: 8px; color: var(--text-secondary);">
                <span style="display: inline-block; width: 8px; height: 8px; background: var(--accent); border-radius: 50; animation: pulse 1.5s infinite;"></span>
                Thinking...
            </div>
            <style>@keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.3; } }</style>
            """, unsafe_allow_html=True)

            start_time = time.time()
            try:
                result = st.session_state.pipeline.query(prompt)
                elapsed = time.time() - start_time
                status_placeholder.empty()

                st.markdown(result["answer"])

                if result["sources"]:
                    source_html = "".join([f'<span class="source-pill">{s}</span>' for s in result["sources"]])
                    st.markdown(f'<div style="margin-top: 8px;">{source_html}</div>', unsafe_allow_html=True)

                st.markdown(f'<div style="font-size: 0.75rem; color: var(--text-secondary); margin-top: 4px;">Responded in {elapsed:.1f}s</div>', unsafe_allow_html=True)

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": result["answer"],
                    "sources": result["sources"],
                    "time": elapsed,
                })
                st.session_state.response_time = elapsed
            except Exception as e:
                status_placeholder.empty()
                st.error(f"Error: {str(e)}")
