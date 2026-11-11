"""Embedding model used by the vector store.

Two interchangeable backends produce vectors for
`sentence-transformers/all-MiniLM-L6-v2` (384-dim, mean pooling):

* ``hf``   — sentence-transformers (needs torch/transformers, ~450 MB RSS).
* ``onnx`` — onnxruntime + tokenizers (~180 MB RSS, no torch).

Because both wrap the *same* model with the same pooling, an index built with
either backend is queryable with the other — the Chroma collection does not
need rebuilding when switching.

Select with ``EMBEDDING_BACKEND=onnx`` (recommended on memory-constrained
hosts such as Render's free tier). Defaults to ``hf``.
"""
import os
from functools import lru_cache
from pathlib import Path

import numpy as np

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
BACKEND = os.getenv("EMBEDDING_BACKEND", "hf").strip().lower()
CACHE_DIR = Path(os.getenv("ONNX_CACHE_DIR", Path.home() / ".cache" / "legal-advisor-onnx"))
MAX_TOKENS = 256


class OnnxMiniLMEmbeddings:
    """LangChain-compatible embeddings backed by onnxruntime (no PyTorch)."""

    def __init__(self, repo_id: str = EMBEDDING_MODEL_NAME, cache_dir: Path | None = None):
        import onnxruntime as ort
        from huggingface_hub import hf_hub_download
        from tokenizers import Tokenizer

        cache_dir = Path(cache_dir or CACHE_DIR)
        cache_dir.mkdir(parents=True, exist_ok=True)

        model_path = hf_hub_download(
            repo_id=repo_id,
            filename="onnx/model.onnx",
            cache_dir=str(cache_dir),
        )
        tokenizer_path = hf_hub_download(
            repo_id=repo_id,
            filename="tokenizer.json",
            cache_dir=str(cache_dir),
        )

        self._tokenizer = Tokenizer.from_file(tokenizer_path)
        self._tokenizer.enable_truncation(max_length=MAX_TOKENS)
        self._tokenizer.enable_padding()

        self._session = ort.InferenceSession(
            model_path, providers=["CPUExecutionProvider"]
        )
        self._input_names = {item.name for item in self._session.get_inputs()}

    def _encode(self, texts: list[str]) -> np.ndarray:
        encodings = self._tokenizer.encode_batch([str(t) for t in texts])
        input_ids = np.array([enc.ids for enc in encodings], dtype=np.int64)
        attention_mask = np.array([enc.attention_mask for enc in encodings], dtype=np.int64)

        feeds = {"input_ids": input_ids, "attention_mask": attention_mask}
        if "token_type_ids" in self._input_names:
            feeds["token_type_ids"] = np.zeros_like(input_ids)

        hidden = self._session.run(None, feeds)[0]

        # Mean pooling over real tokens only, matching sentence-transformers.
        mask = attention_mask[..., None].astype(np.float32)
        summed = (hidden * mask).sum(axis=1)
        counts = np.clip(mask.sum(axis=1), 1e-9, None)
        return summed / counts

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._encode(texts).tolist()

    def embed_query(self, text: str) -> list[float]:
        return self._encode([text])[0].tolist()


@lru_cache(maxsize=1)
def get_embedding_model():
    """Return the configured embedding backend (cached process-wide)."""
    if BACKEND == "onnx":
        return OnnxMiniLMEmbeddings()

    from langchain_huggingface import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
