from __future__ import annotations

import uuid
from typing import Callable, List, Dict, Any

from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct


class QdrantStore:
    """
    Qdrant vector store for RAG.
    Compatible with latest qdrant-client (query_points API).
    """

    def __init__(
        self,
        collection_name: str = "documents",
        dim: int = 384,
        host: str = "localhost",
        port: int = 6333,
    ) -> None:
        self.collection_name = collection_name
        self.dim = dim

        self.client = QdrantClient(host=host, port=port)

        # ✅ recreate collection (tránh lỗi schema cũ)
        self.client.recreate_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(
                size=self.dim,
                distance=Distance.COSINE,
            ),
        )

    # =========================
    # 🔹 ADD DOCUMENTS
    # =========================
    def add_documents(
        self,
        docs: List[Dict[str, Any]],
        embedding_fn: Callable[[str], List[float]],
    ) -> None:
        points = []

        for doc in docs:
            vector = embedding_fn(doc["content"])

            points.append(
                PointStruct(
                    id=str(uuid.uuid4()),  
                    vector=vector,
                    payload={
                        "content": doc["content"],
                        "metadata": doc.get("metadata", {}),
                    },
                )
            )

        self.client.upsert(
            collection_name=self.collection_name,
            points=points,
        )

    # =========================
    # 🔹 SEARCH (FIX LỖI CHÍNH)
    # =========================
    def search(
        self,
        query: str,
        embedding_fn: Callable[[str], List[float]],
        top_k: int = 3,
    ) -> List[Dict[str, Any]]:
        query_vec = embedding_fn(query)

        # ✅ FIX: dùng API mới
        results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vec,
            limit=top_k,
        ).points

        return [
            {
                "content": r.payload.get("content", ""),
                "score": r.score,
                "metadata": r.payload.get("metadata", {}),
            }
            for r in results
        ]

    # =========================
    # 🔹 SEARCH WITH FILTER
    # =========================
    def search_with_filter(
        self,
        query: str,
        embedding_fn: Callable[[str], List[float]],
        top_k: int = 3,
        metadata_filter: dict | None = None,
    ) -> List[Dict[str, Any]]:
        query_vec = embedding_fn(query)

        if metadata_filter:
            results = self.client.query_points(
                collection_name=self.collection_name,
                query=query_vec,
                limit=top_k,
                query_filter={
                    "must": [
                        {
                            "key": f"metadata.{k}",
                            "match": {"value": v},
                        }
                        for k, v in metadata_filter.items()
                    ]
                },
            ).points
        else:
            results = self.client.query_points(
                collection_name=self.collection_name,
                query=query_vec,
                limit=top_k,
            ).points

        return [
            {
                "content": r.payload.get("content", ""),
                "score": r.score,
                "metadata": r.payload.get("metadata", {}),
            }
            for r in results
        ]

    # =========================
    # 🔹 SIZE
    # =========================
    def get_collection_size(self) -> int:
        info = self.client.get_collection(self.collection_name)
        return info.points_count or 0

    # =========================
    # 🔹 DELETE BY METADATA
    # =========================
    def delete_by_metadata(self, key: str, value: str) -> None:
        self.client.delete(
            collection_name=self.collection_name,
            points_selector={
                "filter": {
                    "must": [
                        {
                            "key": f"metadata.{key}",
                            "match": {"value": value},
                        }
                    ]
                }
            },
        )