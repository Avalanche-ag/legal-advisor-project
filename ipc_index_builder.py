"""Build the FAISS index over the Indian Penal Code (IPC) sections.

Run once (and again only if the IPC dataset changes):

    python ipc_index_builder.py
"""
import json
import os

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from vector_database import get_embedding_model

IPC_JSON_PATH = "data/ipc.json"
IPC_FAISS_PATH = "vectorstore/ipc_db"


def load_ipc_data(path: str = IPC_JSON_PATH) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def prepare_documents(ipc_data: list[dict]) -> list[Document]:
    """One IPC section -> one Document (sections are already atomic, no chunking needed)."""
    docs: list[Document] = []
    for entry in ipc_data:
        page_content = (
            f"Section {entry['Section']}: {entry['section_title']}\n\n"
            f"{entry['section_desc']}"
        )
        metadata = {
            "chapter": entry["chapter"],
            "chapter_title": entry["chapter_title"],
            "section": entry["Section"],
            "section_title": entry["section_title"],
        }
        docs.append(Document(page_content=page_content, metadata=metadata))
    return docs


def build_ipc_index(persist_path: str = IPC_FAISS_PATH) -> int:
    data = load_ipc_data()
    docs = prepare_documents(data)
    embedding_model = get_embedding_model()

    os.makedirs(os.path.dirname(persist_path), exist_ok=True)
    db = FAISS.from_documents(docs, embedding_model)
    db.save_local(persist_path)

    print(f"Indexed {len(docs)} IPC sections -> {persist_path}")
    return len(docs)


if __name__ == "__main__":
    build_ipc_index()