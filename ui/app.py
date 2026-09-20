import streamlit as st
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import SCRAPE_SOURCES, CHROMA_PERSIST_DIR
from rag.pipeline import RAGPipeline
from scraper.devops_docs import DevOpsDocScraper
from embeddings.local_embeddings import LocalEmbeddings
from vectorstore.chromadb_store import ChromaVectorStore
from config import EMBEDDING_MODEL, CHROMA_COLLECTION

st.set_page_config(page_title="DevOps RAG Assistant", page_icon="", layout="wide")
st.title("DevOps RAG Assistant")
st.markdown("Ask questions about Kubernetes, Docker, CI/CD, Terraform, monitoring, and more.")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "pipeline" not in st.session_state:
    st.session_state.pipeline = None

with st.sidebar:
    st.header("API Configuration")

    provider = st.selectbox(
        "LLM Provider",
        options=["openrouter", "openai", "google"],
        format_func=lambda x: {
            "openrouter": "OpenRouter",
            "openai": "OpenAI",
            "google": "Google AI Studio"
        }[x],
    )

    if provider == "openrouter":
        api_key = st.text_input("OpenRouter API Key", type="password", help="Get from https://openrouter.ai")
        default_model = "qwen/qwen3.8-27b:free"
    elif provider == "openai":
        api_key = st.text_input("OpenAI API Key", type="password", help="Get from https://platform.openai.com")
        default_model = "gpt-3.5-turbo"
    else:
        api_key = st.text_input("Google AI Studio API Key", type="password", help="Get from https://aistudio.google.com")
        default_model = "gemini-2.0-flash"

    model = st.text_input("Model", value=default_model)

    if st.button("Initialize Pipeline", type="primary"):
        if not api_key:
            st.error("Please enter an API key.")
        else:
            with st.spinner("Loading embedding model and initializing pipeline..."):
                st.session_state.pipeline = RAGPipeline(api_key=api_key, model=model, provider=provider)
                st.success(f"Pipeline initialized with {provider}!")

    st.divider()
    st.header("Data Ingestion")
    selected_sources = st.multiselect(
        "Select documentation sources",
        options=list(SCRAPE_SOURCES.keys()),
        default=list(SCRAPE_SOURCES.keys()),
    )

    if st.button("Ingest Documents"):
        if not selected_sources:
            st.warning("Please select at least one source.")
        else:
            sources_to_scrape = {k: v for k, v in SCRAPE_SOURCES.items() if k in selected_sources}
            progress_text = st.empty()
            progress_bar = st.progress(0)

            scraper = DevOpsDocScraper()
            all_docs = scraper.scrape_all(sources_to_scrape)

            if all_docs:
                progress_text.text("Loading embedding model...")
                embedder = LocalEmbeddings(EMBEDDING_MODEL)
                progress_bar.progress(50)

                texts = [doc.content for doc in all_docs]
                metadatas = [doc.metadata for doc in all_docs]

                progress_text.text("Generating embeddings...")
                embeddings = embedder.embed_documents(texts)
                progress_bar.progress(75)

                progress_text.text("Storing in ChromaDB...")
                store = ChromaVectorStore(persist_dir=CHROMA_PERSIST_DIR, collection_name=CHROMA_COLLECTION)
                store.add_documents(texts, embeddings, metadatas)
                progress_bar.progress(100)

                st.success(f"Done! Ingested {len(all_docs)} documents from {len(selected_sources)} sources.")
            else:
                st.error("No documents were scraped.")

    st.divider()
    st.header("Vector Store Stats")
    try:
        store = ChromaVectorStore(persist_dir=CHROMA_PERSIST_DIR, collection_name=CHROMA_COLLECTION)
        stats = store.get_stats()
        st.metric("Total Documents", stats["total_documents"])
    except Exception:
        st.info("Vector store not initialized yet.")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "sources" in msg and msg["sources"]:
            with st.expander("Sources"):
                for src in msg["sources"]:
                    st.write(f"- {src}")

if prompt := st.chat_input("Ask a DevOps question..."):
    if not st.session_state.pipeline:
        st.error("Please initialize the pipeline first from the sidebar.")
    else:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    result = st.session_state.pipeline.query(prompt)
                    st.markdown(result["answer"])
                    if result["sources"]:
                        with st.expander("Sources"):
                            for src in result["sources"]:
                                st.write(f"- {src}")
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": result["answer"],
                        "sources": result["sources"],
                    })
                except Exception as e:
                    st.error(f"Error: {str(e)}")
