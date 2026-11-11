"""Build the vector index over the IPC dataset.

Two interchangeable formats are supported (``--format``):

* ``numpy``  — single ``.npz`` file (vectors + metadata). No ChromaDB, no
               server, ~200 MB less RAM. Recommended for small corpora and
               memory-constrained hosts (Render free tier).
* ``chroma`` — persistent ChromaDB collection (HNSW).
* ``both``   — default; writes both so either backend can be used.

Usage:
    python scripts/build_index.py [--format numpy|chroma|both]
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))

import numpy as np  # noqa: E402

from app.services.embeddings import get_embedding_model  # noqa: E402

IPC_JSON_PATH = BASE_DIR / "data" / "ipc.json"
VECTORSTORE_DIR = BASE_DIR / "vectorstore"
IPC_CHROMA_PATH = VECTORSTORE_DIR / "ipc_chroma"
IPC_NUMPY_PATH = VECTORSTORE_DIR / "ipc_index.npz"
IPC_COLLECTION = "ipc_sections"


def load_ipc_data(path: Path = IPC_JSON_PATH) -> list[dict]:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def prepare_records(ipc_data: list[dict]) -> list[dict]:
    """One record per IPC section, with the text that gets embedded."""
    records = []
    for entry in ipc_data:
        records.append(
            {
                "section": entry["Section"],
                "section_title": entry["section_title"],
                "chapter": entry["chapter"],
                "chapter_title": entry["chapter_title"],
                "content": entry["section_desc"],
                "text": (
                    f"Section {entry['Section']}: {entry['section_title']}\n\n"
                    f"{entry['section_desc']}"
                ),
            }
        )
    return records


def build_numpy_index(records: list[dict], path: Path = IPC_NUMPY_PATH) -> int:
    model = get_embedding_model()
    vectors = np.asarray(
        model.embed_documents([record["text"] for record in records]),
        dtype=np.float32,
    )

    metadata = [
        {key: value for key, value in record.items() if key != "text"}
        for record in records
    ]

    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        vectors=vectors,
        metadata=np.array(json.dumps(metadata, ensure_ascii=False)),
    )
    print(f"Indexed {len(records)} IPC sections -> {path} {vectors.shape}")
    return len(records)


def build_chroma_index(records: list[dict], path: Path = IPC_CHROMA_PATH) -> int:
    from langchain_chroma import Chroma
    from langchain_core.documents import Document

    documents = [
        Document(
            page_content=record["text"],
            metadata={
                "chapter": record["chapter"],
                "chapter_title": record["chapter_title"],
                "section": record["section"],
                "section_title": record["section_title"],
            },
        )
        for record in records
    ]

    # Rebuild from scratch so re-runs never append to a stale collection.
    if path.is_dir():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)

    Chroma.from_documents(
        documents=documents,
        embedding=get_embedding_model(),
        collection_name=IPC_COLLECTION,
        persist_directory=str(path),
    )
    print(f"Indexed {len(records)} IPC sections -> {path} (chroma:{IPC_COLLECTION})")
    return len(records)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--format", choices=["numpy", "chroma", "both"], default="both")
    args = parser.parse_args()

    records = prepare_records(load_ipc_data())

    if args.format in ("numpy", "both"):
        build_numpy_index(records)
    if args.format in ("chroma", "both"):
        build_chroma_index(records)
    return 0


if __name__ == "__main__":
    sys.exit(main())
