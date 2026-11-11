"""IPC section retrieval backed by ChromaDB."""
from pathlib import Path

from langchain_chroma import Chroma

from .embeddings import get_embedding_model

BASE_DIR = Path(__file__).resolve().parents[2]
IPC_CHROMA_PATH = str(BASE_DIR / "vectorstore" / "ipc_chroma")
IPC_COLLECTION = "ipc_sections"


def retrieve_ipc_sections(query: str, k: int = 3) -> list[dict]:
    """Return the top-k IPC sections most relevant to `query`.

    Each result is a dict with keys: section, section_title, chapter,
    chapter_title, content.
    """
    db = Chroma(
        collection_name=IPC_COLLECTION,
        persist_directory=IPC_CHROMA_PATH,
        embedding_function=get_embedding_model(),
    )
    docs = db.similarity_search(query, k=k)

    return [
        {
            "section": doc.metadata.get("section"),
            "section_title": doc.metadata.get("section_title"),
            "chapter": doc.metadata.get("chapter"),
            "chapter_title": doc.metadata.get("chapter_title"),
            "content": doc.page_content,
        }
        for doc in docs
    ]
