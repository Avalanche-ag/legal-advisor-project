"""IPC section retrieval.

Two interchangeable vector backends (``VECTOR_BACKEND``):

* ``numpy``  — in-memory cosine similarity over ``vectorstore/ipc_index.npz``.
               No ChromaDB import, ~200 MB less RAM. Recommended for 575-section
               corpora and memory-constrained hosts.
* ``chroma`` — persistent ChromaDB collection (HNSW). Default for local dev.
"""
import json
import os
from functools import lru_cache
from pathlib import Path

import numpy as np

from .embeddings import get_embedding_model

BASE_DIR = Path(__file__).resolve().parents[2]
VECTORSTORE_DIR = BASE_DIR / "vectorstore"
IPC_CHROMA_PATH = str(VECTORSTORE_DIR / "ipc_chroma")
IPC_COLLECTION = "ipc_sections"
IPC_NUMPY_PATH = VECTORSTORE_DIR / "ipc_index.npz"

BACKEND = os.getenv("VECTOR_BACKEND", "numpy").strip().lower()

FIELDS = ("section", "section_title", "chapter", "chapter_title", "content")


@lru_cache(maxsize=1)
def _load_numpy_index():
    """Load vectors + metadata once per process."""
    data = np.load(IPC_NUMPY_PATH, allow_pickle=False)
    vectors = data["vectors"].astype(np.float32)
    metadata = json.loads(str(data["metadata"].item()))
    norms = np.linalg.norm(vectors, axis=1) + 1e-9
    return vectors, norms, metadata


def _retrieve_numpy(query: str, k: int) -> list[dict]:
    vectors, norms, metadata = _load_numpy_index()

    query_vector = np.asarray(get_embedding_model().embed_query(query), dtype=np.float32)
    query_norm = np.linalg.norm(query_vector) + 1e-9

    scores = (vectors @ query_vector) / (norms * query_norm)
    top_indices = np.argsort(-scores)[:k]

    return [{field: metadata[i].get(field) for field in FIELDS} for i in top_indices]


def _retrieve_chroma(query: str, k: int) -> list[dict]:
    from langchain_chroma import Chroma

    db = Chroma(
        collection_name=IPC_COLLECTION,
        persist_directory=IPC_CHROMA_PATH,
        embedding_function=get_embedding_model(),
    )
    docs = db.similarity_search(query, k=k)
    return [{field: doc.metadata.get(field) if field != "content" else doc.page_content
             for field in FIELDS} for doc in docs]


def retrieve_ipc_sections(query: str, k: int = 3) -> list[dict]:
    """Return the top-k IPC sections most relevant to `query`.

    Each result is a dict with keys: section, section_title, chapter,
    chapter_title, content.
    """
    if BACKEND == "numpy":
        return _retrieve_numpy(query, k)
    return _retrieve_chroma(query, k)
