import os

from .models import Document


def load_markdown_folder(folder_path: str) -> list[Document]:
    docs = []

    for filename in os.listdir(folder_path):
        if not filename.endswith(".md"):
            continue

        path = os.path.join(folder_path, filename)

        with open(path, "r", encoding="utf-8") as f:
            content = f.read()

        docs.append(
            Document(
                id=filename,
                content=content,
                metadata={"source": filename}
            )
        )

    return docs