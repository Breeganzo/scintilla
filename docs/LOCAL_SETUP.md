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

Postgres stores paper metadata, the ingestion run ledger and evaluation
results. Homebrew runs it as a background service.

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

Then set the connection string in `.env`. Note there is **no password** and the
username is your macOS account name:

```bash
DATABASE_URL=postgres://$(whoami)@localhost:5432/scintilla
```

Useful commands:

```bash
brew services stop postgresql@16       # stop
brew services restart postgresql@16    # restart
psql scintilla                          # open a shell (\dt lists tables, \q quits)
```

---

## 3. OpenSearch

OpenSearch provides BM25 keyword search and kNN vector search in one engine.
It needs Java, which is why Java 17 is in the prerequisites.

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
unassigned because there is nowhere to put them. Not a problem.

**If `brew install opensearch` is unavailable**, use the tarball instead:

```bash
curl -LO https://artifacts.opensearch.org/releases/bundle/opensearch/2.18.0/opensearch-2.18.0-darwin-arm64.tar.gz
tar -xzf opensearch-2.18.0-darwin-arm64.tar.gz
cd opensearch-2.18.0
export OPENSEARCH_INITIAL_ADMIN_PASSWORD='' DISABLE_SECURITY_PLUGIN=true
./opensearch-tar-install.sh
```

That runs in the foreground - leave the terminal open.

---

## 4. Airflow

Added in Phase 2. Airflow runs natively without Docker in standalone mode:

```bash
pip install "apache-airflow==2.10.4"
export AIRFLOW_HOME="$(pwd)/airflow"
airflow standalone
```

It prints an admin password on first run and serves the UI at
`http://localhost:8080`.

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

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `could not connect to server` | Postgres not running | `brew services start postgresql@16` |
| `role "scintilla" does not exist` | `DATABASE_URL` uses the container username | Use `$(whoami)` instead |
| `Connection refused` on :9200 | OpenSearch still starting, or crashed | Wait 30s, then `brew services list`. If `error`, check `/opt/homebrew/var/log/opensearch/` |
| OpenSearch will not start | Heap larger than available RAM | Confirm the `-Xmx512m` line landed in `jvm.options` |
| `command not found: psql` | Versioned formula not on PATH | Re-run the `export PATH` line from step 2 |
| `ModuleNotFoundError` | Virtual environment not active | `source .venv/bin/activate` |
