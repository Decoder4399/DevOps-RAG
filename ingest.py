import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from scraper.devops_docs import DevOpsDocScraper
from embeddings.local_embeddings import LocalEmbeddings
from vectorstore.chromadb_store import ChromaVectorStore
from config import (
    SCRAPE_SOURCES,
    EMBEDDING_MODEL,
    CHROMA_PERSIST_DIR,
    CHROMA_COLLECTION,
)


def main():
    print("=" * 60)
    print("DevOps RAG - Document Ingestion")
    print("=" * 60)

    print("\nStep 1/4: Scraping documentation...")
    scraper = DevOpsDocScraper()
    all_docs = scraper.scrape_all(SCRAPE_SOURCES)

    if not all_docs:
        print("No documents scraped. Exiting.")
        return

    print(f"\nStep 2/4: Loading embedding model ({EMBEDDING_MODEL})...")
    embedder = LocalEmbeddings(EMBEDDING_MODEL)

    texts = [doc.content for doc in all_docs]
    metadatas = [doc.metadata for doc in all_docs]

    print("\nStep 3/4: Generating embeddings...")
    embeddings = embedder.embed_documents(texts)

    print(f"\nStep 4/4: Storing in ChromaDB ({CHROMA_PERSIST_DIR})...")
    store = ChromaVectorStore(persist_dir=CHROMA_PERSIST_DIR, collection_name=CHROMA_COLLECTION)
    store.add_documents(texts, embeddings, metadatas)

    stats = store.get_stats()
    print("\n" + "=" * 60)
    print("Ingestion Complete!")
    print(f"  Total documents: {stats['total_documents']}")
    print(f"  Storage: {stats['persist_directory']}")
    print("=" * 60)
    print("\nRun 'streamlit run ui/app.py' to start the web interface.")


if __name__ == "__main__":
    main()
