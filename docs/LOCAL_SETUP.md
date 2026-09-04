# Local setup without Docker

For machines where Docker is not available. Services are installed natively
with Homebrew instead of run as containers.

Everything works this way - ingestion, retrieval, evaluation, the API and the
frontend. The only thing you cannot do locally is exercise the container
images themselves, and CI does that on every push.

---

## What you need

| Tool | Check | If missing |
|---|---|---|
| Homebrew | `brew --version` | https://brew.sh |
| Python 3.13 | `python3 --version` | `brew install python@3.13` |
| Node 20+ | `node --version` | `brew install node` |
| Java 17+ | `java -version` | `brew install openjdk@17` |
| Git | `git --version` | Ships with Xcode command line tools |

---

## 1. Python environment

A virtual environment keeps this project's packages separate from every other
project on the machine. Without one, two projects needing different Django
versions cannot coexist.

```bash
cd scintilla
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements/dev.txt
```

The prompt gains a `(.venv)` prefix when it is active. Re-run the `source` line
in every new terminal.

---

## 2. PostgreSQL

Postgres stores paper metadata, the ingestion run ledger, evaluation results
**and the embedding vectors** - the dense half of retrieval runs on the
`pgvector` extension. See section 2b and `ingestion/vectors.py` for why the
vectors are here rather than in OpenSearch.

```bash
brew install postgresql@16
brew services start postgresql@16

# Homebrew keeps versioned formulae off the default PATH.
echo 'export PATH="/opt/homebrew/opt/postgresql@16/bin:$PATH"' >> ~/.zshrc
source ~/.zshrc

psql --version    # confirm
```

Create the database. Homebrew's Postgres creates a superuser named after your
macOS account, so no password is needed locally:

```bash
createdb scintilla
createdb scintilla_test

psql -l | grep scintilla    # confirm both exist
```

Then set the connection string in `.env`. There is **no password** and the
username is your macOS account name. Find it with:

```bash
whoami
```

Put the **literal** name into `.env`:

```
DATABASE_URL=postgres://your-macos-username@localhost:5432/scintilla
```

> A `.env` file is read by `python-dotenv`, not by a shell. It does **not**
> expand `$(whoami)`, `$HOME` or any other substitution. Writing
> `postgres://$(whoami)@...` there produces a connection attempt for a role
> literally named `$(whoami)`, which fails with `role does not exist`.

Useful commands:

```bash
brew services stop postgresql@16       # stop
brew services restart postgresql@16    # restart
psql scintilla                          # open a shell (\dt lists tables, \q quits)
```

---

## 2b. pgvector

Dense retrieval needs the `vector` extension. **Do not `brew install pgvector`.**
The bottle is compiled against whichever Postgres major version Homebrew
considers current, which is not `postgresql@16`, so the extension files land in
the wrong `lib` directory and `CREATE EXTENSION vector` fails with
`could not open extension control file`. Build it against the Postgres you are
actually running:

```bash
mkdir -p /tmp/pgvector-build && cd /tmp/pgvector-build
curl -L https://github.com/pgvector/pgvector/archive/refs/tags/v0.8.6.tar.gz | tar -xz
cd pgvector-0.8.6

# This is the line that matters: it points the build at postgresql@16's
# headers and install paths rather than the default major version.
export PG_CONFIG=/opt/homebrew/opt/postgresql@16/bin/pg_config
make
make install       # no sudo needed; Homebrew owns these directories
```

Enable it in both databases and confirm:

```bash
psql scintilla      -c 'CREATE EXTENSION IF NOT EXISTS vector;'
psql scintilla_test -c 'CREATE EXTENSION IF NOT EXISTS vector;'

psql scintilla -c "SELECT extversion FROM pg_extension WHERE extname='vector';"
# 0.8.6

psql scintilla -c "SELECT '[1,2,3]'::vector <=> '[1,2,4]'::vector AS cosine_distance;"
# 0.00853986601633272
```

The migrations also run `CREATE EXTENSION`, so this is belt and braces - but
doing it by hand first means a failure is reported here, with a clear message,
rather than in the middle of `migrate`.

---

## 3. OpenSearch

OpenSearch provides **BM25 keyword search only**. Vector search does *not* run
here - it runs in Postgres via pgvector. This is not a preference, it is forced:

- OpenSearch publishes **no macOS build**. Every `*-darwin-*` URL under
  `artifacts.opensearch.org` returns HTTP 403; only `linux` and `windows`
  artifacts exist. Homebrew works around this by compiling the *min*
  distribution from source, which ships **zero plugins**.
- The `opensearch-knn` plugin is a native library with JNI bindings built only
  for Linux and Windows. `opensearch-plugin install opensearch-knn` fails with
  `Unknown plugin opensearch-knn`, and adding `index.knn` to a mapping fails
  with `unknown setting [index.knn]`.
- Docker would sidestep this by running the Linux image, but Docker is not
  available on the development machine.

So the index built by `manage.py index_papers` contains **no vector field** and
**no `index.knn` setting**. If you have seen an older revision of this file that
suggested otherwise, it was wrong.

OpenSearch needs Java, which is why Java 17 is in the prerequisites.

```bash
brew install opensearch
```

Before starting it, cap the heap. OpenSearch defaults to a quarter of system
RAM, which is more than this project needs and enough to make a laptop
unpleasant:

```bash
echo "-Xms512m" >> /opt/homebrew/etc/opensearch/jvm.options
echo "-Xmx512m" >> /opt/homebrew/etc/opensearch/jvm.options
```

Start it and confirm:

```bash
brew services start opensearch

# Wait a few seconds, then:
curl http://localhost:9200
```

A JSON response containing `"cluster_name"` means it is up.

```bash
curl http://localhost:9200/_cluster/health?pretty    # should be green or yellow
```

Yellow is normal and expected on a single node - it means replicas are
unassigned because there is nowhere to put them. Not a problem. This project
asks for zero replicas, so a single node reports **green**.

> **There is no tarball fallback on macOS.** An earlier revision of this guide
> pointed at `opensearch-2.18.0-darwin-arm64.tar.gz`. That file does not exist
> and never did; the URL returns 403 because the S3 bucket denies `ListBucket`,
> which makes a missing key look like a permissions error rather than a 404.
> If `brew install opensearch` fails, there is no second route on macOS - use
> a Linux machine or the Docker image (see `docs/DOCKER_SETUP.md`).

---

## 3b. The embedding model

Embeddings need PyTorch, which is a large download and is deliberately kept out
of `requirements/base.txt` so that CI and the ingestion tests do not pay for it:

```bash
pip install -r requirements/ml.txt
```

The model itself (`BAAI/bge-small-en-v1.5`, roughly 130 MB) downloads from
Hugging Face on first use and is cached in `~/.cache/huggingface`. The first
`index_papers` run is therefore slower than every subsequent one.

---

## 4. Airflow

Airflow orchestrates the daily harvest. It runs natively, without Docker.

### 4a. Its own virtualenv

Airflow pins a large dependency tree of its own. Installing it next to Django
means neither can be upgraded without negotiating with the other, and a
resolver conflict in the orchestrator would block work on the application it is
only supposed to be triggering. So it gets a separate interpreter:

```bash
python3 -m venv .venv-airflow
./.venv-airflow/bin/pip install --upgrade pip

AIRFLOW_VERSION=3.3.1
PYTHON_VERSION="$(./.venv-airflow/bin/python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
./.venv-airflow/bin/pip install "apache-airflow[postgres]==${AIRFLOW_VERSION}" \
  --constraint "https://raw.githubusercontent.com/apache/airflow/constraints-${AIRFLOW_VERSION}/constraints-${PYTHON_VERSION}.txt"
```

The constraints file is not optional. Airflow has hundreds of transitive
dependencies and pip's resolver will happily pick a combination that imports
but does not run. The constraints file is the set the Airflow team actually
tested against that Python version.

> **Why 3.3.1 and not 2.x?** The original plan said 2.10.4. There is no
> 2.10.4 wheel for Python 3.13 — the 2.x line tops out below the version of
> Python this project targets. Pinning an older Python purely to run an
> end-of-life scheduler is the wrong trade.

### 4b. Its own metadata database

Airflow writes constantly: task instances, heartbeats, XComs. Keeping that out
of the application database means `pg_dump scintilla` is the corpus and nothing
else, and scheduler churn never shows up in application query plans.

```bash
createdb scintilla_airflow
```

### 4c. Environment

Everything Airflow needs is in one sourceable script, so the scheduler and a
human running the same command by hand agree on every value:

```bash
source scripts/airflow_env.sh
```

This is a shell script rather than a `.env` file on purpose: it has to expand
`$(whoami)` and `$PWD`, and python-dotenv reads `.env` literally — it would
hand Postgres a role called `$(whoami)`.

### 4d. Initialise and run

```bash
airflow db migrate      # creates Airflow's tables in scintilla_airflow
airflow standalone      # api-server + scheduler + dag-processor + triggerer
```

The UI is at `http://localhost:8080`. The generated admin password is written
to `airflow/simple_auth_manager_passwords.json.generated`, which is gitignored.

To run the pipeline once without a scheduler — useful for testing:

```bash
airflow dags test arxiv_ingest
```

### 4e. What the DAG does

```
preflight ──▶ harvest[category] ──▶ index ──▶ verify
```

`preflight` fails fast if the venv, `manage.py`, `DATABASE_URL` or OpenSearch
are missing, so a misconfiguration surfaces in two seconds rather than after a
twenty-minute harvest. `harvest` is one mapped task per arXiv category, so one
category rate-limiting does not fail the others. `index` embeds and indexes
only what changed. `verify` asserts the three invariants that matter — no
unembedded chunks, index document count equal to chunk count, every paper
marked indexed — and fails the run if any of them is false.

Tasks shell out to the management commands rather than importing Django. That
keeps one interface, not two: the scheduled path and the manual path are the
same path. The cost is that failures arrive as exit codes and logs rather than
tracebacks, which is acceptable because both commands raise `CommandError` on
failure and exit non-zero.

The categories and per-category limit come from `ARXIV_CATEGORIES` and
`ARXIV_DAILY_LIMIT`, read at DAG parse time.

---

## 5. Configure and run

```bash
cp .env.example .env
```

Edit `.env` and set at minimum:

```bash
DJANGO_SECRET_KEY=<generate one, see below>
DATABASE_URL=postgres://<your-macos-username>@localhost:5432/scintilla
OPENSEARCH_URL=http://localhost:9200
```

Generate a secret key:

```bash
python -c "from django.core.management.utils import get_random_secret_key as k; print(k())"
```

Then:

```bash
python manage.py migrate
python manage.py runserver
```

- API root: http://localhost:8000/api/
- Schema: http://localhost:8000/api/docs/
- Admin: http://localhost:8000/admin/ (after `python manage.py createsuperuser`)

---

## 6. Tests

```bash
pytest                    # unit tests, no services needed
pytest -m integration     # needs Postgres and OpenSearch running
ruff check .              # lint
ruff format .             # format
```

### Which database do the tests use?

`config/settings/test.py` uses `DATABASE_URL` if it is set and falls back to
in-memory SQLite if it is not. SQLite is convenient but forgiving - it accepts
things PostgreSQL rejects, so a green SQLite run is weaker evidence than a
green PostgreSQL run.

To run the suite against the real engine locally:

```bash
DATABASE_URL=postgres://$(whoami)@localhost:5432/scintilla pytest -q
```

Django creates and drops a throwaway `test_scintilla` database for the run, so
your development data is never touched. CI always runs this way against a
PostgreSQL 16 service container - **CI is the authority.**

---

## Everyday commands

```bash
# Start the day
brew services start postgresql@16
brew services start opensearch
source .venv/bin/activate
python manage.py runserver

# End the day
brew services stop opensearch        # frees ~1 GB of RAM
brew services stop postgresql@16
```

`brew services list` shows what is currently running.

### Building the corpus

```bash
# Fetch papers into Postgres. Safe to re-run: unchanged abstracts cost nothing.
python manage.py ingest_arxiv --categories hep-ex,hep-th,cs.IR --limit 1000

# Embed into pgvector and index into OpenSearch, in one pass.
python manage.py index_papers

# Run it again. It should report seen=0 - everything is already current.
python manage.py index_papers
```

`index_papers` only touches chunks that have no vector or no index timestamp.
To force everything, use `--all`. To rebuild into a fresh index and switch the
`papers` alias over atomically when it is finished, use `--rebuild`; the old
index is deliberately left behind so you can verify the new one before deleting
it.

A full pass over ~3,000 abstracts on CPU takes roughly two minutes and peaks
around 1.2 GB of resident memory, almost all of it PyTorch.

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `could not connect to server` | Postgres not running | `brew services start postgresql@16` |
| `role "scintilla" does not exist` | `DATABASE_URL` uses the Docker Compose username, or a literal `$(whoami)` that was never expanded | Run `whoami` and paste the result into `.env` |
| `Connection refused` on :9200 | OpenSearch still starting, or crashed | Wait 30s, then `brew services list`. If `error`, check `/opt/homebrew/var/log/opensearch/` |
| OpenSearch will not start | Heap larger than available RAM | Confirm the `-Xmx512m` line landed in `jvm.options` |
| `command not found: psql` | Versioned formula not on PATH | Re-run the `export PATH` line from step 2 |
| `psql` worked, then stopped working | Re-running `source .venv/bin/activate` calls `deactivate` first, which restores the PATH from **before** the venv was active and drops the `postgresql@16` entry | Activate the venv **first**, then export the PATH. Or open a new shell so `~/.zshrc` applies |
| `runserver` exits with `OperationalError` | Not a bug. `runserver` runs a migration-consistency check before binding the port, so it refuses to start without a database | Start Postgres. To observe the app's own degraded behaviour instead, boot it the way production does: `gunicorn config.wsgi:application --bind 127.0.0.1:8000` |
| `ModuleNotFoundError` | Virtual environment not active | `source .venv/bin/activate` |
| `could not open extension control file ".../vector.control"` | pgvector was installed against a different Postgres major version | Rebuild from source with `PG_CONFIG` pointing at `postgresql@16` - see section 2b |
| `unknown setting [index.knn]` | Something is trying to create a vector field in OpenSearch | The k-NN plugin does not exist on macOS and is not used. Vectors belong in pgvector - see section 3 |
| `ERROR: Unknown plugin opensearch-knn` | Same cause | Same answer. Do not chase this; there is no macOS build |
| `sentence-transformers is not installed` | `requirements/ml.txt` was skipped | `pip install -r requirements/ml.txt` |
| First `index_papers` run hangs for a minute | Downloading the 130 MB model from Hugging Face | Wait. Subsequent runs read from `~/.cache/huggingface` |
| `airflow standalone` starts, then dies with `FileNotFoundError: 'airflow'` | `standalone` does not run the sub-components in-process. It spawns them by executing the string `airflow`, resolved against `PATH`. Calling `./.venv-airflow/bin/airflow` finds the parent by path but the children by name | `source scripts/airflow_env.sh`, which puts `.venv-airflow/bin` on `PATH` |
| `airflow dags list` prints `No data found` right after `db migrate` | The DAG table is written by the dag-processor, not by the CLI. Nothing has parsed the folder yet | `airflow dags reserialize`, or just start `airflow standalone` and wait for the processor's first pass |
| Postgres refuses to start: `FATAL: could not access file "timescaledb"` | `shared_preload_libraries` in `postgresql.conf` names an extension that is not built for this Postgres major version. The server cannot start at all | Comment the line out in `/opt/homebrew/var/postgresql@16/postgresql.conf` and restart. Check what is actually installed with `ls /opt/homebrew/opt/postgresql@16/lib/postgresql/` |
| `brew services` shows `postgresql@16  error 1` and restarts every few seconds | An orphaned postmaster still holds `postmaster.pid`, so launchd's restart always loses the race. The real error is hidden underneath | `kill` the stale PID named in the lock file, then `brew services start postgresql@16` and read the log again |
