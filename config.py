import os
from dotenv import load_dotenv

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")

LLM_MODEL = "liquid/lfm-2.5-2.6b:free"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

CHROMA_PERSIST_DIR = "./chroma_db"
CHROMA_COLLECTION = "devops_docs"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
TOP_K_RESULTS = 5

SCRAPE_SOURCES = {
    "kubernetes": [
        "https://kubernetes.io/docs/concepts/",
        "https://kubernetes.io/docs/tutorials/",
        "https://kubernetes.io/docs/reference/kubectl/",
    ],
    "docker": [
        "https://docs.docker.com/get-started/",
        "https://docs.docker.com/compose/",
    ],
    "aws": [
        "https://docs.aws.amazon.com/cli/latest/reference/",
    ],
    "terraform": [
        "https://developer.hashicorp.com/terraform/docs",
        "https://developer.hashicorp.com/terraform/language",
    ],
    "github_actions": [
        "https://docs.github.com/en/actions",
        "https://docs.github.com/en/actions/using-workflows",
        "https://docs.github.com/en/actions/using-jobs",
    ],
    "prometheus": [
        "https://prometheus.io/docs/prometheus/latest/getting_started/",
        "https://prometheus.io/docs/prometheus/latest/querying/basics/",
    ],
}
