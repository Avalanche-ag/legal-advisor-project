"""IPC section retrieval — plain function (no CrewAI/@tool)."""
from langchain_community.vectorstores import FAISS

from vector_database import get_embedding_model

IPC_FAISS_PATH = "vectorstore/ipc_db"


def retrieve_ipc_sections(query: str, k: int = 3) -> list[dict]:
    """Return the top-k IPC sections most relevant to `query`.

    Each result is a dict with keys: section, section_title, chapter,
    chapter_title, content (the section's full text).
    """
    embedding_model = get_embedding_model()
    db = FAISS.load_local(
        IPC_FAISS_PATH,
        embedding_model,
        allow_dangerous_deserialization=True,
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