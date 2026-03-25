from __future__ import annotations

from typing import Iterable

import chromadb
from chromadb.api.models.Collection import Collection
from sentence_transformers import SentenceTransformer

from backend.app.services.chunking import Chunk


class VectorStoreService:
    def __init__(self, persist_dir: str, collection_name: str) -> None:
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.collection: Collection = self.client.get_or_create_collection(name=collection_name)

    def add_chunks(self, chunks: Iterable[Chunk]) -> int:
        chunk_list = list(chunks)
        if not chunk_list:
            return 0

        documents = [chunk.text for chunk in chunk_list]
        ids = [chunk.chunk_id for chunk in chunk_list]
        metadatas = [chunk.metadata | {'source': chunk.source} for chunk in chunk_list]
        embeddings = self.embedding_model.encode(documents, convert_to_numpy=True).tolist()

        self.collection.upsert(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=embeddings,
        )
        return len(chunk_list)

    def similarity_search(self, query: str, top_k: int = 4) -> list[dict]:
        query_embedding = self.embedding_model.encode([query], convert_to_numpy=True).tolist()
        result = self.collection.query(
            query_embeddings=query_embedding,
            n_results=top_k,
            include=['documents', 'metadatas', 'distances'],
        )

        hits: list[dict] = []
        documents = result.get('documents', [[]])[0]
        metadatas = result.get('metadatas', [[]])[0]
        distances = result.get('distances', [[]])[0]
        ids = result.get('ids', [[]])[0]

        for idx, doc in enumerate(documents):
            hits.append(
                {
                    'chunk_id': ids[idx],
                    'text': doc,
                    'metadata': metadatas[idx],
                    'distance': distances[idx],
                }
            )
        return hits
