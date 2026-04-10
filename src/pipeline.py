from .loader import load_markdown_folder
from .chunking import RecursiveChunker
from .store_qdrant import QdrantStore
from .embeddings import LocalEmbedder


def build_vector_db(data_path: str):
    docs = load_markdown_folder(data_path)

    chunker = RecursiveChunker(chunk_size=500)
    embedder = LocalEmbedder()  # hoặc MockEmbedder()

    store = QdrantStore(dim=384)

    all_chunks = []

    for doc in docs:
        chunks = chunker.chunk(doc.content)

        for chunk in chunks:
            all_chunks.append({
                "content": chunk,
                "metadata": doc.metadata
            })

    store.add_documents(all_chunks, embedder)

    return store, embedder