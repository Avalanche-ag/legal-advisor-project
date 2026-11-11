"""Build the Chroma vector index over the IPC dataset.

Usage:
    python scripts/build_index.py
"""
import json
import shutil
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))

from langchain_chroma import Chroma  # noqa: E402
from langchain_core.documents import Document  # noqa: E402

from app.services.embeddings import get_embedding_model  # noqa: E402

IPC_JSON_PATH = BASE_DIR / "data" / "ipc.json"
IPC_CHROMA_PATH = BASE_DIR / "vectorstore" / "ipc_chroma"
IPC_COLLECTION = "ipc_sections"


def load_ipc_data(path: Path = IPC_JSON_PATH) -> list[dict]:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def prepare_documents(ipc_data: list[dict]) -> list[Document]:
    """One IPC section -> one Document (sections are already atomic)."""
    docs: list[Document] = []
    for entry in ipc_data:
        docs.append(
            Document(
                page_content=(
                    f"Section {entry['Section']}: {entry['section_title']}\n\n"
                    f"{entry['section_desc']}"
                ),
                metadata={
                    "chapter": entry["chapter"],
                    "chapter_title": entry["chapter_title"],
                    "section": entry["Section"],
                    "section_title": entry["section_title"],
                },
            )
        )
    return docs


def build_ipc_index(persist_path: str | Path = IPC_CHROMA_PATH) -> int:
    persist_path = Path(persist_path)
    docs = prepare_documents(load_ipc_data())

    # Rebuild from scratch so re-runs never append to a stale collection.
    if persist_path.is_dir():
        shutil.rmtree(persist_path)
    persist_path.mkdir(parents=True, exist_ok=True)

    Chroma.from_documents(
        documents=docs,
        embedding=get_embedding_model(),
        collection_name=IPC_COLLECTION,
        persist_directory=str(persist_path),
    )
    print(f"Indexed {len(docs)} IPC sections -> {persist_path}")
    return len(docs)


if __name__ == "__main__":
    build_ipc_index()
