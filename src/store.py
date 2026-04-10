from __future__ import annotations

from typing import Any, Callable

from .chunking import _dot
from .embeddings import _mock_embed
from .models import Document


class EmbeddingStore:
    def __init__(
        self,
        collection_name: str = "documents",
        embedding_fn: Callable[[str], list[float]] | None = None,
    ) -> None:
        self._embedding_fn = embedding_fn or _mock_embed
        self._collection_name = collection_name
        self._use_chroma = False
        self._store: list[dict[str, Any]] = []
        self._collection = None
        self._next_index = 0

        try:
            import chromadb

            client = chromadb.Client()

            # ✅ FIX: reset collection (quan trọng để test size = 0)
            try:
                client.delete_collection(name=collection_name)
            except Exception:
                pass

            self._collection = client.get_or_create_collection(name=collection_name)
            self._use_chroma = True

        except Exception:
            self._use_chroma = False
            self._collection = None

    # =========================
    # CREATE RECORD
    # =========================
    def _make_record(self, doc: Document) -> dict[str, Any]:
        embedding = self._embedding_fn(doc.content)

        metadata = doc.metadata.copy() if doc.metadata else {}
        metadata["doc_id"] = doc.id

        record = {
            "id": str(self._next_index),
            "content": doc.content,
            "embedding": embedding,
            "metadata": metadata,
        }

        # ✅ QUAN TRỌNG NHẤT
        self._next_index += 1
        return record

    # =========================
    # IN-MEMORY SEARCH
    # =========================
    def _search_records(self, query: str, records, top_k: int):
        query_vec = self._embedding_fn(query)

        scored = []
        for r in records:
            score = _dot(query_vec, r["embedding"])
            scored.append((score, r))

        # ✅ sort descending
        scored.sort(key=lambda x: x[0], reverse=True)

        return [
            {
                "score": score,
                "content": r["content"],
                "metadata": r["metadata"],
            }
            for score, r in scored[:top_k]
        ]

    # =========================
    # ADD DOCUMENTS
    # =========================
    def add_documents(self, docs: list[Document]) -> None:
        if self._use_chroma:
            ids, texts, embeddings, metadatas = [], [], [], []

            for doc in docs:
                record = self._make_record(doc)

                ids.append(record["id"])
                texts.append(record["content"])
                embeddings.append(record["embedding"])

                # ✅ metadata không được rỗng
                meta = record["metadata"] if record["metadata"] else {"dummy": "1"}
                metadatas.append(meta)

            self._collection.add(
                ids=ids,
                documents=texts,
                embeddings=embeddings,
                metadatas=metadatas,
            )

        else:
            for doc in docs:
                self._store.append(self._make_record(doc))

    # =========================
    # SEARCH
    # =========================
    def search(self, query: str, top_k: int = 5):
        if self._use_chroma:
            results = self._collection.query(
                query_embeddings=[self._embedding_fn(query)],
                n_results=top_k,
            )

            docs = results.get("documents", [[]])[0]
            metas = results.get("metadatas", [[]])[0]
            scores = results.get("distances", [[]])[0]

            # ✅ FIX: đảo dấu để sort đúng
            return [
                {
                    "score": -score,
                    "content": doc,
                    "metadata": meta,
                }
                for doc, meta, score in zip(docs, metas, scores)
            ]

        else:
            return self._search_records(query, self._store, top_k)

    # =========================
    # SIZE
    # =========================
    def get_collection_size(self) -> int:
        if self._use_chroma:
            return self._collection.count()
        return len(self._store)

    # =========================
    # SEARCH WITH FILTER
    # =========================
    def search_with_filter(self, query: str, top_k: int = 3, metadata_filter: dict = None):
        if not metadata_filter:
            return self.search(query, top_k)

        if self._use_chroma:
            results = self._collection.query(
                query_embeddings=[self._embedding_fn(query)],
                n_results=top_k,
                where=metadata_filter,
            )

            docs = results.get("documents", [[]])[0]
            metas = results.get("metadatas", [[]])[0]
            scores = results.get("distances", [[]])[0]

            return [
                {
                    "score": -score,
                    "content": doc,
                    "metadata": meta,
                }
                for doc, meta, score in zip(docs, metas, scores)
            ]

        else:
            filtered = []

            for r in self._store:
                match = True
                for k, v in metadata_filter.items():
                    if r["metadata"].get(k) != v:
                        match = False
                        break
                if match:
                    filtered.append(r)

            return self._search_records(query, filtered, top_k)

    # =========================
    # DELETE DOCUMENT
    # =========================
    def delete_document(self, doc_id: str) -> bool:
        if self._use_chroma:
            results = self._collection.get()

            ids = results["ids"]
            metas = results["metadatas"]

            to_delete = [
                id_
                for id_, meta in zip(ids, metas)
                if meta.get("doc_id") == doc_id
            ]

            if not to_delete:
                return False

            self._collection.delete(ids=to_delete)
            return True

        else:
            before = len(self._store)

            self._store = [
                r for r in self._store
                if r["metadata"].get("doc_id") != doc_id
            ]

            return len(self._store) < before