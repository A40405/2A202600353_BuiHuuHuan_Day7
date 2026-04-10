from src.pipeline import build_vector_db
from src.agent import KnowledgeBaseAgent


DATA_PATH = r"D:\AI_thucchien\assignments-main\day7\Day-07-Lab-Data-Foundations\data\data2"


def main():
    store, embedder = build_vector_db(DATA_PATH)

    # wrap store cho agent
    class Wrapper:
        def __init__(self, store, embedder):
            self.store = store
            self.embedder = embedder

        def search(self, query, top_k=3):
            return self.store.search(query, self.embedder, top_k)

    agent = KnowledgeBaseAgent(Wrapper(store, embedder))

    while True:
        q = input("\n❓ Question: ")
        if q.lower() == "exit":
            break

        answer = agent.answer(q)
        print("\n🤖 Answer:", answer)


if __name__ == "__main__":
    main()