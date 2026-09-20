import chromadb
from chromadb.config import Settings
from typing import List, Dict, Tuple
import os


class ChromaVectorStore:
    def __init__(
        self,
        persist_dir: str = "./chroma_db",
        collection_name: str = "devops_docs",
    ):
        self.persist_dir = persist_dir
        os.makedirs(persist_dir, exist_ok=True)

        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        print(f"ChromaDB initialized. Collection '{collection_name}' has {self.collection.count()} documents.")

    def add_documents(
        self,
        texts: List[str],
        embeddings: List[List[float]],
        metadatas: List[Dict[str, str]],
        batch_size: int = 100,
    ) -> int:
        total_added = 0
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            batch_embeddings = embeddings[i : i + batch_size]
            batch_metadatas = metadatas[i : i + batch_size]
            batch_ids = [f"doc_{i + j}" for j in range(len(batch_texts))]

            self.collection.upsert(
                ids=batch_ids,
                documents=batch_texts,
                embeddings=batch_embeddings,
                metadatas=batch_metadatas,
            )
            total_added += len(batch_texts)
            print(f"  Upserted batch {i // batch_size + 1}: {total_added}/{len(texts)} documents")

        return total_added

    def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
    ) -> List[Tuple[str, Dict[str, str], float]]:
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        matches = []
        if results["documents"] and results["documents"][0]:
            for doc, meta, dist in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            ):
                similarity = 1 - dist
                matches.append((doc, meta, similarity))

        return matches

    def get_stats(self) -> Dict:
        return {
            "total_documents": self.collection.count(),
            "persist_directory": self.persist_dir,
        }

    def clear(self):
        self.client.delete_collection(self.collection.name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection.name,
            metadata={"hnsw:space": "cosine"},
        )
        print("Collection cleared.")
