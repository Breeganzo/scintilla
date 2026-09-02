# Scintilla

Hybrid search and grounded question answering over scientific preprints, with a
retrieval evaluation harness that runs in CI.

Most retrieval systems are shipped without ever being measured. This one exists
to measure itself: keyword search, vector search, and the fusion of the two are
compared against a hand-built golden set, and a regression gate fails the build
if retrieval quality drops.

---

## Status

Built in the open over 14 days. Honest markers - nothing is ticked before it
works.

| Phase | Scope | State |
|---|---|---|
| 1 · Foundation | Django, PostgreSQL, containers, CI | 🟡 In progress |
| 2 · Ingestion | arXiv harvesting, embeddings, OpenSearch, Airflow | ⬜ Not started |
| 3 · Retrieval & evaluation | BM25, dense, RRF, golden set, metrics, CI gate | ⬜ Not started |
| 4 · Ship | React frontend, MCP server, deployment, hardening | ⬜ Not started |

**No evaluation numbers are published yet.** When they exist they will appear
here, including the cases where hybrid retrieval performs *worse* than its
components.

---

## The idea

Keyword search and vector search fail differently.

**BM25** matches words. Ask it for a detector name or a dataset identifier and it
is close to unbeatable. Ask it for a concept phrased in words the author did not
use and it returns nothing useful.

**Dense retrieval** matches meaning. It handles paraphrase and conceptual queries
well, and it will confidently miss an exact identifier because the embedding does
not preserve rare tokens.

Combining them with Reciprocal Rank Fusion is standard practice. What is not
standard is checking whether the combination actually helped - and on which kinds
of query it hurt. That check is the point of this project.

---

## Architecture

```
arXiv API
    │
    ▼
Airflow DAG ──► parse ──► embed (BGE-small) ──► OpenSearch
    │                                              │  BM25 + kNN
    ▼                                              │
PostgreSQL  ◄─────────────────────────────────────┘
 metadata,                    │
 run ledger,                  ▼
 eval results        Reciprocal Rank Fusion
                              │
                              ▼
                     grounded answer + citations
                              │
                    ┌─────────┴─────────┐
                    ▼                   ▼
              REST API (DRF)       MCP server
                    │                   │
                    ▼                   ▼
             React frontend      LLM tool-calling host
```

Full diagrams: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

---

## Stack

| Layer | Choice | Why |
|---|---|---|
| API | Django 5 + Django REST Framework | Migrations, admin and auth without assembling them by hand |
| Metadata | PostgreSQL 16 | Relational integrity for the run ledger and evaluation history |
| Index | OpenSearch 2 | BM25 and kNN vector search in one engine, Apache-2.0 |
| Embeddings | BAAI/bge-small-en-v1.5 | 384-dim, runs on CPU, small enough for a free-tier VM |
| Orchestration | Apache Airflow | Retries, backfill and run history that cron cannot express |
| Generation | Llama 3.3 70B via Groq | Free tier; the system degrades to search-only without it |
| Frontend | React 18 + TypeScript + Vite | Strict-mode types generated from the OpenAPI schema |
| Protocol | Model Context Protocol | Exposes retrieval as callable tools |

Every component is free to run. See [docs/TECH_STACK.md](docs/TECH_STACK.md).

---

## Running it

Two supported paths.

**With Docker** - one command, everything included:

```bash
cp .env.example .env
docker compose up --build
```

**Without Docker** - services installed natively via Homebrew, for machines
where Docker is not available:

See [docs/LOCAL_SETUP.md](docs/LOCAL_SETUP.md).

Either way the API is then at `http://localhost:8000/api/` and the schema at
`http://localhost:8000/api/docs/`.

---

## Tests

```bash
pytest                      # unit tests, no external services needed
pytest -m integration       # requires Postgres and OpenSearch running
ruff check .                # lint
```

CI runs the unit suite, the linter, a dependency audit and a container build on
every push.

---

## Licence

MIT - see [LICENSE](LICENSE).
