from functools import lru_cache

from langchain_huggingface import HuggingFaceEmbeddings

# Single embedding model used by the IPC vector store.
# Kept consistent between index-building and retrieval so vectors stay comparable.
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


@lru_cache(maxsize=1)
def get_embedding_model() -> HuggingFaceEmbeddings:
    """Lazy singleton for the HuggingFace embedding model."""
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)