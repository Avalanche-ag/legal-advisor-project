"""Embedding model used by the vector store.

Backends (``EMBEDDING_BACKEND``):

* ``hf``   — sentence-transformers / PyTorch (large, ~250-330 MB RSS).
* ``onnx`` — onnxruntime + tokenizers (light, ~180 MB RSS, no torch).

Both wrap ``sentence-transformers/all-MiniLM-L6-v2`` with **mean pooling and no
L2 normalisation**, producing 384-dim float32 vectors. The same
``vectorstore/ipc_index.npz`` is therefore valid for either backend — vectors
differ only by float32 rounding (cosine similarity > 0.999 in practice).

Extra env knobs (matter when PyTorch is in play):

* ``TORCH_NUM_THREADS``  — call ``torch.set_num_threads(n)``; ``1`` minimises
                           per-thread arena memory (recommended under 512 MB).
* ``TOKENIZERS_PARALLELISM`` — defaults to ``false`` to avoid the tokenizer
                           thread pool.
"""
import os
from functools import lru_cache
from pathlib import Path

import numpy as np

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
BACKEND = os.getenv("EMBEDDING_BACKEND", "hf").strip().lower()
CACHE_DIR = Path(os.getenv("ONNX_CACHE_DIR", Path.home() / ".cache" / "legal-advisor-onnx"))
MAX_TOKENS = 256
TORCH_NUM_THREADS = int(os.getenv("TORCH_NUM_THREADS", "0"))


class OnnxMiniLMEmbeddings:
    """LangChain-compatible embeddings backed by onnxruntime (no PyTorch)."""

    def __init__(self, repo_id: str = EMBEDDING_MODEL_NAME, cache_dir: Path | None = None):
        import onnxruntime as ort
        from huggingface_hub import hf_hub_download
        from tokenizers import Tokenizer

        cache_dir = Path(cache_dir or CACHE_DIR)
        cache_dir.mkdir(parents=True, exist_ok=True)

        model_path = hf_hub_download(
            repo_id=repo_id, filename="onnx/model.onnx", cache_dir=str(cache_dir)
        )
        tokenizer_path = hf_hub_download(
            repo_id=repo_id, filename="tokenizer.json", cache_dir=str(cache_dir)
        )

        self._tokenizer = Tokenizer.from_file(tokenizer_path)
        self._tokenizer.enable_truncation(max_length=MAX_TOKENS)
        self._tokenizer.enable_padding()

        self._session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        self._input_names = {item.name for item in self._session.get_inputs()}

    def _encode(self, texts: list[str]) -> np.ndarray:
        encodings = self._tokenizer.encode_batch([str(t) for t in texts])
        input_ids = np.array([enc.ids for enc in encodings], dtype=np.int64)
        attention_mask = np.array([enc.attention_mask for enc in encodings], dtype=np.int64)

        feeds = {"input_ids": input_ids, "attention_mask": attention_mask}
        if "token_type_ids" in self._input_names:
            feeds["token_type_ids"] = np.zeros_like(input_ids)

        hidden = self._session.run(None, feeds)[0]

        mask = attention_mask[..., None].astype(np.float32)
        summed = (hidden * mask).sum(axis=1)
        counts = np.clip(mask.sum(axis=1), 1e-9, None)
        return summed / counts

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._encode(texts).tolist()

    def embed_query(self, text: str) -> list[float]:
        return self._encode([text])[0].tolist()


class TorchMiniLMEmbeddings:
    """sentence-transformers (PyTorch) backend.

    Imports torch/transformers lazily and never touches langchain, ChromaDB or
    gRPC, so selecting ``hf`` cannot pull the vector-store or LLM client stacks.
    """

    def __init__(self, model_name: str = EMBEDDING_MODEL_NAME):
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

        import torch

        if TORCH_NUM_THREADS > 0:
            torch.set_num_threads(TORCH_NUM_THREADS)

        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_name)
        self.dimension = self._model.get_sentence_embedding_dimension()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors = self._model.encode(
            [str(t) for t in texts],
            batch_size=32,
            convert_to_numpy=True,
            normalize_embeddings=False,  # must match the ONNX backend / index
            show_progress_bar=False,
        )
        return np.asarray(vectors, dtype=np.float32).tolist()

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


@lru_cache(maxsize=1)
def get_embedding_model():
    """Return the configured embedding backend (cached process-wide)."""
    if BACKEND == "onnx":
        return OnnxMiniLMEmbeddings()
    return TorchMiniLMEmbeddings()
