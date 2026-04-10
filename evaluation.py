from collections import Counter
from typing import List, Dict

from src.pipeline import build_vector_db
from src.agent import KnowledgeBaseAgent


# =========================
# 🔹 TEST QUERIES (3 bệnh)
# =========================
TEST_QUERIES = [
    {
        "question": "Amip ăn não là gì?",
        "expected_doc": "amip",
        "gold_answer_keywords": ["amip", "não", "ký sinh"]
    },
    {
        "question": "Triệu chứng áp xe gan là gì?",
        "expected_doc": "ap-xe-gan",
        "gold_answer_keywords": ["đau", "gan", "sốt"]
    },
    {
        "question": "Alzheimer là bệnh gì?",
        "expected_doc": "alzheimer",
        "gold_answer_keywords": ["trí nhớ", "thoái hóa", "não"]
    }
]


# =========================
# 🔹 RETRIEVAL PRECISION
# =========================
def evaluate_retrieval(store, embedder, top_k=3):
    print("\n=== 🔍 RETRIEVAL PRECISION ===")

    total_score = 0

    for q in TEST_QUERIES:
        results = store.search(q["question"], embedder, top_k)

        relevant = 0
        for r in results:
            content = r["content"].lower()
            if q["expected_doc"] in content:
                relevant += 1

        score = 2 if relevant >= 2 else (1 if relevant == 1 else 0)
        total_score += score

        print(f"\nQ: {q['question']}")
        print(f"Relevant in top-{top_k}: {relevant}")
        print(f"Score: {score}/2")

    print(f"\n👉 TOTAL RETRIEVAL SCORE: {total_score}/{len(TEST_QUERIES)*2}")


# =========================
# 🔹 CHUNK COHERENCE
# =========================
def evaluate_chunking(docs, chunker):
    print("\n=== ✂️ CHUNK COHERENCE ===")

    all_chunks = []

    for doc in docs:
        chunks = chunker.chunk(doc.content)
        all_chunks.extend(chunks)

    lengths = [len(c) for c in all_chunks]

    print(f"Total chunks: {len(all_chunks)}")
    print(f"Avg length: {sum(lengths)/len(lengths):.2f}")
    print(f"Min length: {min(lengths)}")
    print(f"Max length: {max(lengths)}")

    print("👉 Đánh giá thủ công: chunk có bị cắt giữa câu không?")


# =========================
# 🔹 METADATA UTILITY
# =========================
def evaluate_filter(store, embedder):
    print("\n=== 🧠 METADATA UTILITY ===")

    query = "Triệu chứng Alzheimer"

    no_filter = store.search(query, embedder, top_k=3)
    with_filter = store.search_with_filter(
        query,
        embedder,
        top_k=3,
        metadata_filter={"source": "alzheimer"}
    )

    print("\n--- Without Filter ---")
    for r in no_filter:
        print(r["content"][:100])

    print("\n--- With Filter ---")
    for r in with_filter:
        print(r["content"][:100])


# =========================
# 🔹 GROUNDING QUALITY
# =========================
def evaluate_grounding(agent):
    print("\n=== 🤖 GROUNDING QUALITY ===")

    for q in TEST_QUERIES:
        answer = agent.answer(q["question"]).lower()

        match = any(k in answer for k in q["gold_answer_keywords"])

        print(f"\nQ: {q['question']}")
        print(f"Answer: {answer[:150]}...")
        print(f"Match keyword: {match}")


# =========================
# 🔹 SCORE DISTRIBUTION
# =========================
def evaluate_score_distribution(store, embedder):
    print("\n=== 📊 SCORE DISTRIBUTION ===")

    query = "Alzheimer là gì?"
    results = store.search(query, embedder, top_k=5)

    scores = [r["score"] for r in results]

    print("Scores:", scores)

    diff = max(scores) - min(scores)
    print("Score spread:", diff)


# =========================
# 🔹 MAIN EVALUATION
# =========================
def run_evaluation(data_path):
    store, embedder = build_vector_db(data_path)

    # wrap store cho agent
    class Wrapper:
        def __init__(self, store, embedder):
            self.store = store
            self.embedder = embedder

        def search(self, query, top_k=3):
            return self.store.search(query, self.embedder, top_k)

    agent = KnowledgeBaseAgent(Wrapper(store, embedder))

    # load docs lại để chunk eval
    from src.loader import load_markdown_folder
    from src.chunking import RecursiveChunker

    docs = load_markdown_folder(data_path)
    chunker = RecursiveChunker(chunk_size=500)

    # chạy từng metric
    evaluate_retrieval(store, embedder)
    evaluate_chunking(docs, chunker)
    evaluate_filter(store, embedder)
    evaluate_grounding(agent)
    evaluate_score_distribution(store, embedder)


# =========================
# 🔹 RUN
# =========================
if __name__ == "__main__":
    DATA_PATH = r"D:\AI_thucchien\assignments-main\day7\Day-07-Lab-Data-Foundations\data\data2"
    run_evaluation(DATA_PATH)