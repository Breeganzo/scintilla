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
| 1 · Foundation | Django, PostgreSQL, containers, CI | ✅ Complete |
| 2 · Ingestion | arXiv harvesting, embeddings, indexing, Airflow | 🔄 Airflow outstanding |
| 3 · Retrieval & evaluation | BM25, dense, RRF, golden set, metrics, CI gate | ⬜ Not started |
| 4 · Ship | React frontend, MCP server, deployment, hardening | ⬜ Not started |

Phase 1 means the foundation is verified, not that the system does anything
useful yet: a clean clone installs, migrates against PostgreSQL 16, serves a
read-only API and passes 29 tests, and CI checks lint, tests, dependencies and
the container build on every push.

Phase 2 currently harvests arXiv into PostgreSQL, chunks against the embedding
model's own tokenizer, embeds into pgvector and indexes into OpenSearch. The
corpus is 2,975 papers across `hep-ex`, `hep-th` and `cs.IR`, and re-running
either command indexes nothing, which is the property that makes a scheduled
pipeline safe. Orchestrating that with Airflow is the remaining piece.

**Nothing is fused or measured yet.** There is no Reciprocal Rank Fusion, no
golden set and no retrieval metrics, so no claim is made about retrieval
quality. That is Phase 3, and it is the part of this project that matters most.

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
Airflow DAG ──► parse ──► chunk ──► embed (BGE-small)
    │                                    │
    │                    ┌───────────────┴───────────────┐
    │                    ▼                               ▼
    │            OpenSearch (BM25)          PostgreSQL + pgvector (kNN)
    │             lexical index              metadata, run ledger,
    │                    │                   eval results, vectors
    ▼                    │                               │
  ledger                 └───────────────┬───────────────┘
                                         ▼
                              Reciprocal Rank Fusion
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

The two indexes are split on purpose. OpenSearch publishes no macOS build and
its k-NN plugin has native libraries only for Linux and Windows, so vector
search could not run there on the development machine. Putting the vectors in
Postgres turned out to be the better design anyway: a chunk and its embedding
are written in one transaction, the deployment target runs one JVM instead of a
larger one, and Reciprocal Rank Fusion combines *ranked lists*, so it does not
care that the two runs came from different systems. The trade-off given up is
real and worth stating: OpenSearch can filter a k-NN query by category *inside*
the vector search, whereas here a filtered dense query is a SQL `WHERE` applied
after the HNSW scan. `ingestion/vectors.py` records this in full.

Full diagrams are published alongside the retrieval evaluation in Phase 3.

---

## Stack

| Layer | Choice | Why |
|---|---|---|
| API | Django 5 + Django REST Framework | Migrations, admin and auth without assembling them by hand |
| Metadata | PostgreSQL 16 | Relational integrity for the run ledger and evaluation history |
| Lexical index | OpenSearch 2 | BM25, Apache-2.0, the reference implementation of the algorithm |
| Vector index | pgvector (HNSW, cosine) | Vectors live beside the rows they describe, written in one transaction |
| Embeddings | BAAI/bge-small-en-v1.5 | 384-dim, runs on CPU, small enough for a free-tier VM |
| Orchestration | Apache Airflow | Retries, backfill and run history that cron cannot express |
| Generation | Llama 3.3 70B via Groq | Free tier; the system degrades to search-only without it |
| Frontend | React 18 + TypeScript + Vite | Strict-mode types generated from the OpenAPI schema |
| Protocol | Model Context Protocol | Exposes retrieval as callable tools |

Every component is free to run: PostgreSQL, OpenSearch and Airflow are all
self-hosted and open source, the embedding model runs on CPU, and Groq's free
tier covers generation. The system degrades to search-only if the LLM is
unavailable rather than failing.

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

Without `DATABASE_URL` the suite falls back to in-memory SQLite so it runs
anywhere. To exercise the engine production uses:

```bash
DATABASE_URL=postgres://$(whoami)@localhost:5432/scintilla pytest -q
```

CI runs the unit suite, the linter, a dependency audit and a container build on
every push. CI always uses a real PostgreSQL 16 service container, so a green
CI run is stronger evidence than a green local run.

---

## Licence

MIT - see [LICENSE](LICENSE).
