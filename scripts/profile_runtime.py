"""Profile the configured runtime profile.

Reports peak RSS, retrieval results for known queries, and (optionally) the
numerical agreement between the ONNX and PyTorch embedding backends.

Usage:
    # hybrid profile
    EMBEDDING_BACKEND=hf VECTOR_BACKEND=numpy GENERATION_BACKEND=rest \
        python scripts/profile_runtime.py --ask

    # compare HF vs ONNX vectors for the same text
    EMBEDDING_BACKEND=hf python scripts/profile_runtime.py --compare

    # torch thread tuning
    TORCH_NUM_THREADS=1 EMBEDDING_BACKEND=hf python scripts/profile_runtime.py --ask
"""
import argparse
import os
import resource
import sys
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))

EXPECTED = {
    "punishment for theft": [382, 379, 411],
    "punishment for murder": [303, 302, 307],
}


def peak_mb() -> float:
    """Peak resident set size in MB (ru_maxrss is bytes on macOS, KB on Linux)."""
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return round(value / (1024 * 1024), 1) if sys.platform == "darwin" else round(value / 1024, 1)


def current_mb() -> float:
    """Current RSS in MB from the OS (macOS/Linux)."""
    try:
        import subprocess

        if sys.platform == "darwin":
            out = subprocess.check_output(["/bin/ps", "-o", "rss=", "-p", str(os.getpid())])
            return round(int(out.strip()) / 1024, 1)
        with open("/proc/self/status") as handle:
            for line in handle:
                if line.startswith("VmRSS:"):
                    return round(int(line.split()[1]) / 1024, 1)
    except Exception:
        pass
    return 0.0


def compare_backends() -> int:
    """Cosine agreement between ONNX and PyTorch vectors for the same inputs."""
    from app.services.embeddings import OnnxMiniLMEmbeddings, TorchMiniLMEmbeddings

    import numpy as np

    texts = ["punishment for theft", "Whoever commits theft shall be punished"]
    onnx = OnnxMiniLMEmbeddings()
    torch_ = TorchMiniLMEmbeddings()

    a = np.asarray(onnx.embed_documents(texts), dtype=np.float32)
    b = np.asarray(torch_.embed_documents(texts), dtype=np.float32)

    print(f"onnx shape {a.shape} | torch shape {b.shape}")
    print(f"dimension match: {a.shape == b.shape}")
    for i, text in enumerate(texts):
        cos = float(a[i] @ b[i] / (np.linalg.norm(a[i]) * np.linalg.norm(b[i]) + 1e-12))
        max_diff = float(np.abs(a[i] - b[i]).max())
        print(f"  '{text[:40]}': cosine={cos:.6f} max_abs_diff={max_diff:.5f}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ask", action="store_true", help="run one /ask round-trip")
    parser.add_argument("--compare", action="store_true", help="compare HF vs ONNX vectors")
    parser.add_argument("--query", default="punishment for theft")
    args = parser.parse_args()

    print("=== profile ===")
    print("EMBEDDING_BACKEND   :", os.getenv("EMBEDDING_BACKEND", "hf"))
    print("VECTOR_BACKEND      :", os.getenv("VECTOR_BACKEND", "chroma"))
    print("GENERATION_BACKEND  :", os.getenv("GENERATION_BACKEND", "sdk"))
    print("TORCH_NUM_THREADS   :", os.getenv("TORCH_NUM_THREADS", "0 (torch default)"))

    if args.compare:
        return compare_backends()

    started = time.time()
    from app.services.retriever import retrieve_ipc_sections

    print(f"\nimport + start: {round(time.time() - started, 1)}s | RSS {current_mb()} MB")

    started = time.time()
    results = retrieve_ipc_sections(args.query, 3)
    sections = [row["section"] for row in results]
    print(f"retrieval ({round(time.time() - started, 2)}s): {sections}")

    expected = EXPECTED.get(args.query)
    if expected:
        print(f"expected: {expected} -> {'MATCH' if sections == expected else 'MISMATCH'}")

    if args.ask:
        from app import create_app

        app = create_app()
        client = app.test_client()
        from app.models import User

        with app.app_context():
            if not User.query.filter_by(email="prof@local.dev").first():
                user = User(email="prof@local.dev", username="proftest", role="admin")
                user.set_password("profpass123")
                from app import db

                db.session.add(user)
                db.session.commit()

        client.post("/login", json={"email": "prof@local.dev", "password": "profpass123"})
        started = time.time()
        response = client.post("/ask", json={"query": args.query})
        payload = response.get_json() or {}
        print(
            f"/ask 200={response.status_code} in {round(time.time() - started, 1)}s "
            f"| rss {current_mb()} MB"
        )
        print("answer:", (payload.get("answer") or payload.get("error", ""))[:100])

    print(f"\nPEAK RSS: {peak_mb()} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
