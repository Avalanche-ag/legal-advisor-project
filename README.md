# ⚖️ AI Lawyer — IPC Assistant

A retrieval-augmented generation (RAG) pipeline that answers natural-language
questions about the **Indian Penal Code (IPC)**, with user accounts,
role-based access control, persistent query logging, and an admin dashboard.

Answers are grounded in retrieved IPC sections, so every response cites
specific section numbers — cutting down hallucination and making the system
**auditable**.

---

## Features

| Feature | Details |
|---|---|
| **RAG Q&A** | Top-3 IPC section retrieval → Gemini grounded answer with citations |
| **Authentication** | Session login with hashed passwords (`Flask-Login` + `werkzeug`) |
| **RBAC** | `admin` and `user` roles; admin-only dashboard and APIs |
| **Query logging** | Prompt, response, citations, latency, status and error persisted per request |
| **User history** | Paginated personal query history (`/api/history`) |
| **Admin dashboard** | System metrics, user management, filterable system logs |
| **PDF report** | Export your successful queries as a PDF transcript |
| **Pluggable runtime** | Swap vector store, embedding backend and LLM transport via env vars |

---

## Architecture

```mermaid
flowchart TB
    subgraph Client
        UI["index.html · app.js"]
        ADM["admin.html · admin.js"]
    end

    subgraph Flask["Flask app package"]
        AUTH["auth: /register /login /logout /api/me"]
        MAIN["main: / /ask /report /api/history"]
        ADMIN["admin: /admin/dashboard /api/admin/*"]
        MODELS["models.py: User · QueryLog"]
    end

    subgraph Services
        RAG["services/rag.py"]
        RET["services/retriever.py<br/>numpy | chroma"]
        EMB["services/embeddings.py<br/>onnx | hf"]
        GEN["services/generator.py<br/>rest | sdk"]
    end

    MYSQL[("MySQL<br/>user · query_log")]
    IDX[("Vector index<br/>ipc_index.npz | ipc_chroma")]
    API["Google Gemini API"]

    UI --> MAIN
    ADM --> ADMIN
    AUTH --> MODELS
    MAIN --> MODELS
    ADMIN --> MODELS
    MODELS --> MYSQL
    MAIN --> RAG
    RAG --> RET
    RET --> EMB
    RET --> IDX
    RAG --> GEN
    GEN --> API
```

**Request flow for one question**

1. `POST /ask` (login required) validates the query.
2. `services/rag.answer_ipc_query()` embeds the query and retrieves the top-3
   IPC sections from the vector index.
3. Sections are formatted into the prompt template and sent to Gemini.
4. The answer plus citation payload is written to `query_log` with latency and
   status (`SUCCESS` / `FAILED`).

---

## Runtime profiles

Every heavy dependency has a light alternative, selected with environment
variables. Both variants produce equivalent behaviour.

| Concern | Local / dev (default) | Light (free hosting) | Env var |
|---|---|---|---|
| Vector store | ChromaDB (HNSW) | NumPy `.npz` + cosine | `VECTOR_BACKEND=chroma\|numpy` |
| Embeddings | sentence-transformers (torch) | onnxruntime MiniLM *(optional)* | `EMBEDDING_BACKEND=hf\|onnx` |
| LLM transport | — | plain HTTPS (`requests`) for all providers | always REST |
| Database | SQLite file | MySQL | `MYSQL_*` / `DATABASE_URL` |

Measured peak RSS for a full `/ask` request (same machine, same query):

| Profile | Peak RSS | Notes |
|---|---|---|
| Chroma + torch + gRPC (legacy default) | 562 MB | over 512 MB ceiling |
| Chroma + torch + REST | 548 MB | rest saves ~14 MB |
| Chroma + ONNX + REST | 463 MB | onnx saves ~85 MB vs torch |
| **NumPy + ONNX + REST** (default light) | **301 MB** | numpy saves ~163 MB vs chroma |
| **NumPy + torch + REST** (hybrid) | **~386 MB projected** | derived: 301 + 85 = 386; cross-check 548 − 163 = 385 |

Both embedding backends wrap the same `all-MiniLM-L6-v2` model with mean
pooling and **no L2 normalisation**, so the same `.npz` index serves either
backend. Retrieval uses cosine similarity, which is magnitude-invariant, so
even a normalisation difference could not change the ranking.

Both embedding backends wrap the same `all-MiniLM-L6-v2` model with identical
mean pooling, so **the same index works with either** — retrieval returns the
same sections ([382, 379, 411] for *"punishment for theft"*).

---

### Embeddings with PyTorch (shipped default)

The deployed profile uses the NumPy index with sentence-transformers
embeddings:

```bash
pip install -r requirements.txt      # CPU wheels via --extra-index-url

export VECTOR_BACKEND=numpy
export EMBEDDING_BACKEND=hf
export TORCH_NUM_THREADS=1          # smaller per-thread arenas
export TOKENIZERS_PARALLELISM=false
```

Projected peak ≈386 MB, leaving ~126 MB (25%) under the 512 MB ceiling —
within the 10 % safety policy. Install **CPU-only** torch wheels: the default
Linux wheel bundles CUDA (~2.5 GB installed) and raises RSS.

Cold start is slower than ONNX (PyTorch import ≈3–8 s on Linux vs ≈0.5 s), but
it happens on import of the first request, well inside Gunicorn's 120 s worker
timeout. The `preDeployCommand` builds the index before the service starts.

If RSS ever exceeds ~460 MB, escalate in this order:
`TORCH_NUM_THREADS=1` → `--threads 2` → torch dynamic quantization
(`torch.quantization.quantize_dynamic`, int8) — the last one changes vectors,
so rebuild the index with `--format numpy` afterwards.

## LLM fallback chain

Generation uses three providers in order, all over plain HTTP (`requests`):

| # | Provider | Model (default) | Why | Timeout |
|---|---|---|---|---|
| 1 | Groq | `openai/gpt-oss-20b` | free tier, fastest | 12 s |
| 2 | MiniMax | `abab6.5s` | very low cost, OpenAI-compatible | 15 s |
| 3 | Gemini | `gemini-2.5-flash-lite` | free tier, high availability | 15 s |

Groq's model catalogue changes without notice — check
`GET https://api.groq.com/openai/v1/models` and set `GROQ_MODEL` accordingly
(currently `openai/gpt-oss-20b`, `openai/gpt-oss-120b`, `qwen/qwen3.8-27b`).

A provider is skipped when its API key is absent. Any failure (429, timeout,
5xx, empty completion) logs a warning and moves to the next provider; if all
three fail, the request raises and `/ask` returns a clean 500 while the
`query_log` row records `status=FAILED` with the concatenated reason.

**Where the legal knowledge comes from:** the providers are *language* models —
none of them searches the web or knows the IPC. The answer is grounded by the
local vector index:

```
data/ipc.json  (575 sections, source of truth, build time only)
      │  python scripts/build_index.py
      ▼
vectorstore/ipc_index.npz   (575 × 384 float32 vectors + section text)
      │  cosine similarity, ~2 ms, $0 per query, no network
      ▼
top-3 IPC sections ──► prompt context ──► Groq │ MiniMax │ Gemini
```

So `ipc.json` is read **only when (re)building the index**; at runtime the
`.npz` file supplies the citations, and the LLM merely phrases them.

## Project structure

```
.
├── app/
│   ├── __init__.py          # create_app() factory, db, login manager
│   ├── models.py            # User, QueryLog
│   ├── auth/                # register/login/logout/api-me + @admin_required
│   ├── main/                # chat page, /ask, /report, /api/history
│   ├── admin/               # dashboard page + admin management APIs
│   ├── services/            # embeddings, retriever, generator, RAG, PDF
│   ├── templates/           # base, index, login, register, admin
│   └── static/              # app.js, admin.js, style.css
├── data/ipc.json            # 575 IPC sections (source dataset)
├── instance/                # SQLite fallback (gitignored)
├── notebooks/evaluation.ipynb
├── scripts/
│   ├── build_index.py             # --format numpy|chroma|both
│   ├── init_db.py                 # create tables + seed admin
│   └── verify_rbac.py             # 44 end-to-end checks
├── requirements.txt         # full local stack
├── requirements.txt         # single runtime set (CPU torch wheels)
├── render.yaml
└── wsgi.py
```

---

## Tech stack

| Layer | Technology |
|---|---|
| Backend | Flask 3, Gunicorn |
| Auth | Flask-Login, Werkzeug password hashing |
| Persistence | Flask-SQLAlchemy + MySQL (PyMySQL), SQLite fallback |
| Vector store | ChromaDB, or NumPy cosine index |
| Embeddings | `all-MiniLM-L6-v2` via sentence-transformers (onnxruntime optional) |
| Generation | Groq → MiniMax → Gemini fallback chain (plain HTTP via `requests`) |
| Reporting | ReportLab |
| Frontend | Vanilla HTML/CSS/JS |
| Deploy | Render (`render.yaml`) |

---

## Quickstart (local)

```bash
git clone https://github.com/Avalanche-ag/legal-advisor-project.git
cd legal-advisor-project

python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

cp .env.example .env            # add GEMINI_API_KEY + SECRET_KEY

python scripts/build_index.py   # writes vectorstore/ipc_chroma + ipc_index.npz
python scripts/init_db.py       # SQLite (default) or MySQL if MYSQL_* set

flask --app wsgi run --port 5050
```

Open <http://127.0.0.1:5050>, register a user, and ask a question.

Default admin (override with `ADMIN_EMAIL` / `ADMIN_USERNAME` / `ADMIN_PASSWORD`):

```
email:    admin@local.dev
password: admin12345
```

> Change the admin credentials before exposing the app publicly.

---

## Using MySQL

Set the connection variables (or a single `DATABASE_URL`) and the app switches
database automatically — no code change:

```bash
export MYSQL_HOST=your-host
export MYSQL_PORT=3306
export MYSQL_USER=ipc
export MYSQL_PASSWORD=secret
export MYSQL_DATABASE=ipc_app

python scripts/init_db.py                      # create tables + admin
```

The URI is built as
`mysql+pymysql://user:pass@host:port/db?charset=utf8mb4`, and the engine uses
`pool_pre_ping` + `pool_recycle=280` so idle connections survive managed-DB
restarts.

**Free MySQL hosts** (both work with this project): Aiven (1 GB free),
TiDB Cloud Serverless (MySQL-compatible, 5 GB free). Cracken/PlanetScale no
longer offer a free tier.

If `MYSQL_HOST` and `DATABASE_URL` are both unset, the app falls back to
`instance/ipc_app.db` (SQLite) for local development.

---

## API

| Method | Path | Access | Purpose |
|---|---|---|---|
| `GET` | `/` | public | Chat page |
| `POST` | `/ask` | user | Ask a question (logged) |
| `GET` | `/report` | user | PDF of your successful queries |
| `GET` | `/api/history` | user | Paginated personal history |
| `POST` | `/register` | public | Create a standard user |
| `POST` | `/login` | public | Start a session |
| `GET` | `/logout` | user | End the session |
| `GET` | `/api/me` | user | Current profile |
| `GET` | `/admin/dashboard` | admin | Admin UI |
| `GET` | `/api/admin/users` | admin | All users + query counts |
| `PATCH` | `/api/admin/users/<id>/role` | admin | Promote/demote |
| `PATCH` | `/api/admin/users/<id>/toggle-status` | admin | Activate/deactivate |
| `GET` | `/api/admin/logs` | admin | System logs (`?status=`, `?sort=latency`) |
| `GET` | `/api/admin/stats` | admin | Totals, success rate, avg latency |

Unauthenticated API calls return **401 JSON**; non-admin calls to admin routes
return **403**.

---

## Data model

| Table | Key columns |
|---|---|
| `user` | `id`, `email` (unique), `username` (unique), `password_hash`, `role`, `is_active`, `created_at` |
| `query_log` | `id`, `user_id` → `user.id` (cascade), `query_text`, `response_text`, `retrieved_sections` (JSON), `status`, `error_message`, `latency_ms`, `created_at` |

The vector index is separate storage (`vectorstore/ipc_index.npz` or the
ChromaDB folder) holding 575 embeddings, one per IPC section.

---

## Deployment (Render)

`render.yaml` ships with the lightweight profile:

```yaml
buildCommand:     pip install -r requirements.txt
preDeployCommand: python scripts/build_index.py --format numpy && python scripts/init_db.py
startCommand:     gunicorn wsgi:app --workers 1 --threads 4 --timeout 120 \
                              --max-requests 500 --max-requests-jitter 50
env: VECTOR_BACKEND=numpy, EMBEDDING_BACKEND=hf, GROQ_API_KEY / MINIMAX_API_KEY / GEMINI_API_KEY,
     MALLOC_ARENA_MAX=2, MYSQL_*, GEMINI_API_KEY, SECRET_KEY, ADMIN_*
```

Set `MYSQL_*` to a managed MySQL instance (see above) so users and logs
survive deploys — Render's disk is ephemeral.

**Build note:** `PYTHON_VERSION=3.12.7` is pinned in `render.yaml`. Without it,
Render may select a newer interpreter where `scipy`/`torch` have no binary
wheel, and pip attempts a source build — which fails with
`error while generating package metadata`.

**Memory:** the light profile peaks around 300 MB, leaving ~200 MB of headroom
under the 512 MB free limit. If you switch back to Chroma/torch, upgrade to a
2 GB instance.

> If the service still reports memory exhaustion, check that
> `VECTOR_BACKEND=numpy` and `EMBEDDING_BACKEND=hf` are actually set, and that
> `PYTHON_VERSION=3.12.7` is present (scipy/torch need 3.12 wheels).

---

### First boot and health checks

Tables and the initial admin are created automatically at startup, so a
dashboard-managed service works even if `preDeployCommand` never ran. If
`ADMIN_PASSWORD` is set, an existing admin's password is re-synced to it on
boot — that is the quickest fix for "login failed" after changing the env var.

`GET /healthz` reports the database, whether the vector index is present, and
which provider keys are set — it returns 503 when something is missing:

```json
{"status":"ok","database":"ok","vector_index":"ok",
 "vector_backend":"numpy","embedding_backend":"hf",
 "providers":{"groq":true,"minimax":false,"gemini":true}}
```

> The prebuilt `vectorstore/ipc_index.npz` (884 KB) is committed, so `/ask`
> works immediately after deploy. If it is ever absent the app rebuilds it on
> the first request (`AUTO_BUILD_INDEX=0` disables that).

> If you leave `MYSQL_*` unset on Render, the app uses SQLite on the ephemeral
> disk: accounts and history are wiped on every restart and the admin is
> re-seeded from `ADMIN_*`. For persistence, point it at MySQL.

## Known limitations

- **Single worker** — SQLite sessions and the in-process index cache assume one
  Gunicorn worker; scale vertically (threads) or move to Redis sessions first.
- **IPC-only corpus** — CrPC/BNSS, BSA and judgments are not indexed yet.
- **No rate limiting** — an authenticated user can exhaust the Gemini quota.
- **No password reset / email verification**.
- **No vendor SDKs at runtime** — generation is plain HTTP; richer SDK features
  (function calling, schema-constrained output) are not used.

---

## Roadmap

- [ ] Fine-tuned legal SLM as the primary provider (Gemini stays as fallback)
- [ ] Alembic migrations for schema evolution
- [ ] Expand the corpus (BNS, BNSS, BSA, landmark judgments)
- [ ] Rate limiting, audit export, password reset
