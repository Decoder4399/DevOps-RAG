# DevOps RAG Assistant

A Retrieval-Augmented Generation (RAG) pipeline for DevOps queries. Ask questions about Kubernetes, Docker, CI/CD, Terraform, Prometheus, and more - get answers with source citations.

## Features

- **Multi-Provider LLM Support**: OpenRouter, OpenAI, Google AI Studio
- **Automatic Model Fallback**: If one model is rate-limited, automatically tries the next
- **Free Tier Friendly**: Works with free OpenRouter models (no API cost)
- **Local Embeddings**: Uses `sentence-transformers` (no API cost)
- **Web Scraping**: Fetches docs from official sources (Kubernetes, Docker, Terraform, etc.)
- **Streamlit UI**: Browser-based chat interface with source citations

## Tech Stack

| Component | Choice |
|-----------|--------|
| Embeddings | `all-MiniLM-L6-v2` (local, free) |
| Vector Store | ChromaDB (local, free) |
| LLM | OpenRouter / OpenAI / Google AI Studio |
| UI | Streamlit |
| Framework | Python, httpx |

## Quick Start

### 1. Clone the repo

```bash
git clone https://github.com/Decoder4399/DevOps-RAG.git
cd DevOps-RAG
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
pip install google-genai  # Only if using Google AI Studio
```

### 3. Set up API key

Copy `.env.example` to `.env` and add your key:

```bash
cp .env.example .env
```

Edit `.env`:
```
OPENROUTER_API_KEY=your-key-here
GOOGLE_API_KEY=your-key-here
```

**Get free API keys:**
- **OpenRouter**: https://openrouter.ai (free models available)
- **Google AI Studio**: https://aistudio.google.com (free tier)
- **OpenAI**: https://platform.openai.com (paid)

### 4. Ingest documentation

```bash
python ingest.py
```

This scrapes DevOps docs and stores them in ChromaDB.

### 5. Launch the web UI

```bash
streamlit run ui/app.py
```

Opens at **http://localhost:8501**

## Usage

1. Open http://localhost:8501
2. In the sidebar, select your **LLM Provider** (OpenRouter/OpenAI/Google)
3. Paste your **API Key**
4. Click **Initialize Pipeline**
5. Ask questions in the chat

### Example Questions

- "How do I deploy a Kubernetes app?"
- "What is a Dockerfile?"
- "How to set up a GitHub Actions workflow?"
- "Explain Terraform state management"
- "How to configure Prometheus alerting rules?"

## Project Structure

```
DevOps-RAG/
├── .env.example          # API key template
├── .gitignore
├── config.py             # Configuration
├── requirements.txt      # Dependencies
├── ingest.py             # Data ingestion script
├── scraper/
│   └── devops_docs.py    # Web scraper
├── embeddings/
│   └── local_embeddings.py
├── vectorstore/
│   └── chromadb_store.py
├── rag/
│   └── pipeline.py       # RAG chain with multi-provider support
└── ui/
    └── app.py            # Streamlit web interface
```

## Supported Documentation Sources

| Source | URL |
|--------|-----|
| Kubernetes | kubernetes.io/docs |
| Docker | docs.docker.com |
| AWS | docs.aws.amazon.com |
| Terraform | developer.hashicorp.com/terraform |
| GitHub Actions | docs.github.com/en/actions |
| Prometheus | prometheus.io/docs |

## Free Models (OpenRouter)

The pipeline automatically falls back between these free models:

- `liquid/lfm-2.5-2.6b:free`
- `inclusionai/ling-3.0-flash-sante:free`
- `inclusionai/ling-3.0-flash-vl:free`
- `nex-agi/nex-n2.5-mini:free`
- `dots-studio/dots-3-note-preview:free`

## License

MIT
