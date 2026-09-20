import httpx
import json
import time
from typing import List, Dict, Tuple, Optional, Generator
from embeddings.local_embeddings import LocalEmbeddings
from vectorstore.chromadb_store import ChromaVectorStore
from config import (
    OPENROUTER_API_KEY,
    OPENROUTER_BASE_URL,
    OPENAI_API_KEY,
    GOOGLE_API_KEY,
    LLM_MODEL,
    EMBEDDING_MODEL,
    CHROMA_PERSIST_DIR,
    CHROMA_COLLECTION,
    TOP_K_RESULTS,
)

SYSTEM_PROMPT = """You are a senior DevOps expert assistant. You help engineers with:
- Kubernetes: deployments, services, ingress, troubleshooting, kubectl commands
- Docker: Dockerfiles, Compose, container networking, security
- CI/CD: GitHub Actions, Jenkins, GitLab CI pipelines
- Infrastructure as Code: Terraform, Ansible, CloudFormation
- Monitoring: Prometheus, Grafana, alerting, metrics
- Cloud: AWS, GCP, Azure services and best practices

When answering:
1. Provide specific commands, configurations, or code examples when applicable
2. Explain the reasoning behind your recommendations
3. Include best practices and security considerations
4. If the context doesn't contain enough information, say so honestly
5. Be concise but thorough"""

FREE_MODELS = [
    "liquid/lfm-2.5-2.6b:free",
    "inclusionai/ling-3.0-flash-sante:free",
    "inclusionai/ling-3.0-flash-vl:free",
    "nex-agi/nex-n2.5-mini:free",
    "dots-studio/dots-3-note-preview:free",
]


class RAGPipeline:
    def __init__(self, api_key=None, model=None, provider="openrouter"):
        self.provider = provider
        self.api_key = api_key
        self.model = model

        if provider == "openrouter":
            self.api_key = api_key or OPENROUTER_API_KEY
            self.model = model or LLM_MODEL
        elif provider == "openai":
            self.api_key = api_key or OPENAI_API_KEY
            self.model = model or "gpt-3.5-turbo"
        elif provider == "google":
            self.api_key = api_key or GOOGLE_API_KEY
            self.model = model or "gemini-2.0-flash"

        print("Initializing RAG Pipeline...")
        self.embeddings = LocalEmbeddings(EMBEDDING_MODEL)
        self.vector_store = ChromaVectorStore(
            persist_dir=CHROMA_PERSIST_DIR,
            collection_name=CHROMA_COLLECTION,
        )
        print(f"RAG Pipeline ready (Provider: {self.provider}, Model: {self.model})")

    def _retrieve_context(self, query, top_k=TOP_K_RESULTS):
        query_embedding = self.embeddings.embed_query(query)
        matches = self.vector_store.search(query_embedding, top_k=top_k)
        return [(doc, meta) for doc, meta, score in matches]

    def _build_context(self, retrieved):
        context_parts = []
        for i, (doc, meta) in enumerate(retrieved, 1):
            source = meta.get("source", "unknown")
            url = meta.get("url", "")
            context_parts.append(f"[Source {i}: {source} - {url}]\n{doc}")
        return "\n\n---\n\n".join(context_parts)

    def _try_openrouter_model(self, model, messages, timeout=60.0):
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "http://localhost:8501",
            "X-Title": "DevOps RAG Assistant",
        }
        payload = {"model": model, "messages": messages}

        with httpx.Client(timeout=timeout) as client:
            response = client.post(
                f"{OPENROUTER_BASE_URL}/chat/completions",
                headers=headers,
                json=payload,
            )
            if response.status_code == 200:
                return response.json()["choices"][0]["message"]["content"]
            elif response.status_code == 429:
                raise Exception("429 rate limited")
            else:
                raise Exception(f"HTTP {response.status_code}: {response.text[:200]}")

    def _call_openrouter(self, messages):
        models_to_try = [self.model] + [m for m in FREE_MODELS if m != self.model]

        for model in models_to_try:
            try:
                print(f"  Trying model: {model}")
                result = self._try_openrouter_model(model, messages)
                self.model = model
                return result
            except Exception as e:
                if "429" in str(e):
                    print(f"  {model} rate limited, trying next...")
                    time.sleep(2)
                    continue
                raise

        raise Exception("All models rate limited. Please wait a moment and try again.")

    def _call_openai(self, messages):
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {"model": self.model, "messages": messages}

        with httpx.Client(timeout=120.0) as client:
            response = client.post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers,
                json=payload,
            )
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]

    def _call_google(self, messages):
        from google import genai

        client = genai.Client(api_key=self.api_key)
        contents = []
        for msg in messages:
            role = "user" if msg["role"] in ["user", "system"] else "model"
            contents.append({"role": role, "parts": [msg["content"]]})

        response = client.models.generate_content(model=self.model, contents=contents)
        return response.text

    def _call_llm(self, messages):
        if self.provider == "openrouter":
            return self._call_openrouter(messages)
        elif self.provider == "openai":
            return self._call_openai(messages)
        elif self.provider == "google":
            return self._call_google(messages)
        else:
            raise ValueError(f"Unknown provider: {self.provider}")

    def query(self, user_message, top_k=TOP_K_RESULTS):
        retrieved = self._retrieve_context(user_message, top_k)
        context = self._build_context(retrieved)

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Context from documentation:\n\n{context}\n\n---\n\nQuestion: {user_message}"},
        ]

        answer = self._call_llm(messages)
        sources = list({meta.get("source", "unknown") for _, meta in retrieved})

        return {
            "answer": answer,
            "reasoning": [],
            "sources": sources,
            "retrieved_docs": [(doc, meta) for doc, meta in retrieved],
        }

    def query_stream(self, user_message, top_k=TOP_K_RESULTS):
        retrieved = self._retrieve_context(user_message, top_k)
        context = self._build_context(retrieved)

        sources = list({meta.get("source", "unknown") for _, meta in retrieved})
        yield {"type": "sources", "sources": sources}

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Context from documentation:\n\n{context}\n\n---\n\nQuestion: {user_message}"},
        ]

        if self.provider in ["openrouter", "openai"]:
            base_url = OPENROUTER_BASE_URL if self.provider == "openrouter" else "https://api.openai.com/v1"
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }
            if self.provider == "openrouter":
                headers["HTTP-Referer"] = "http://localhost:8501"
                headers["X-Title"] = "DevOps RAG Assistant"

            models_to_try = [self.model] + [m for m in FREE_MODELS if m != self.model]

            for model in models_to_try:
                payload = {"model": model, "messages": messages, "stream": True}
                try:
                    with httpx.Client(timeout=120.0) as client:
                        with client.stream("POST", f"{base_url}/chat/completions", headers=headers, json=payload) as response:
                            if response.status_code == 429:
                                print(f"  {model} rate limited, trying next...")
                                time.sleep(2)
                                continue
                            response.raise_for_status()
                            self.model = model
                            for line in response.iter_lines():
                                if line.startswith("data: "):
                                    line_data = line[6:]
                                    if line_data.strip() == "[DONE]":
                                        break
                                    try:
                                        chunk = json.loads(line_data)
                                        delta = chunk["choices"][0].get("delta", {})
                                        content = delta.get("content", "")
                                        if content:
                                            yield {"type": "content", "content": content}
                                    except json.JSONDecodeError:
                                        continue
                            return
                except Exception as e:
                    if "429" in str(e):
                        continue
                    raise

            yield {"type": "content", "content": "All models are rate limited. Please wait a moment and try again."}
        else:
            answer = self._call_llm(messages)
            yield {"type": "content", "content": answer}

    def get_stats(self):
        return {
            "provider": self.provider,
            "model": self.model,
            "embedding_model": EMBEDDING_MODEL,
            "vector_store": self.vector_store.get_stats(),
        }
