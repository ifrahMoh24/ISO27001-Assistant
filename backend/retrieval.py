"""Semantic search over the ISO 27001 vector store.

Shared by the API (backend/main.py) and the evaluation script
(eval/evaluate.py), so both measure exactly the same retrieval.
"""

import os
from typing import List

import chromadb
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

from backend.rag_setup import COLLECTION_NAME, DB_PATH, EMBEDDING_MODEL

# Chunks further away than this (cosine distance) are treated as unrelated.
# Chosen from eval/evaluate.py: see the README "Evaluation" section.
MAX_DISTANCE = float(os.getenv("MAX_DISTANCE", "0.80"))

embedding_model = SentenceTransformer(EMBEDDING_MODEL)

_client = chromadb.PersistentClient(path=str(DB_PATH))
try:
    collection = _client.get_collection(name=COLLECTION_NAME)
except Exception as exc:  # the vector store was never built
    raise RuntimeError(
        "Vector store not found. Build it first with: python -m backend.rag_setup"
    ) from exc


class Source(BaseModel):
    id: int
    control: str  # e.g. "A.7.7", or "" for general sections
    title: str
    text: str
    distance: float


def search(query: str, top_k: int, max_distance: float = MAX_DISTANCE) -> List[Source]:
    """Return the top_k closest chunks to the query within max_distance."""
    results = collection.query(
        query_embeddings=embedding_model.encode([query]).tolist(),
        n_results=top_k,
    )
    sources: List[Source] = []
    for doc, meta, dist in zip(
        results["documents"][0], results["metadatas"][0], results["distances"][0]
    ):
        if dist <= max_distance:
            sources.append(
                Source(
                    id=len(sources) + 1,
                    control=meta.get("control", ""),
                    title=meta.get("title", ""),
                    text=doc,
                    distance=round(dist, 3),
                )
            )
    return sources
