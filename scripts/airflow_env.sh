# Airflow environment for local development.
#
#   source scripts/airflow_env.sh
#
# This is a shell script, not a .env file, and the difference matters: a shell
# expands $(whoami) and $PWD, whereas python-dotenv reads .env literally and
# would hand Postgres a role called "$(whoami)".
#
# Airflow lives in its own virtualenv (.venv-airflow) and its own metadata
# database (scintilla_airflow). Both separations are deliberate:
#
#   * Airflow pins a large dependency tree. Installing it alongside Django
#     means neither can be upgraded without negotiating with the other, and a
#     resolver conflict in the orchestrator would block work on the
#     application it is only supposed to be triggering.
#   * Airflow writes to its metadata database constantly - task instances,
#     heartbeats, XComs. Keeping that out of the application database means a
#     `pg_dump scintilla` is the corpus and nothing else, and scheduler churn
#     never appears in application query plans.

set -a

# Homebrew keeps postgresql@16 off the default PATH.
export PATH="/opt/homebrew/opt/postgresql@16/bin:$PATH"

# Resolved once here so every command below agrees on the project root.
SCINTILLA_HOME="${SCINTILLA_HOME:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)}"

# `airflow standalone` does not run the api-server, scheduler and dag-processor
# in-process; it spawns them by executing the string "airflow", resolved
# against PATH. Calling ./.venv-airflow/bin/airflow therefore starts, and then
# immediately dies with FileNotFoundError: 'airflow' - the parent was found by
# path, the children by name. Putting the venv's bin on PATH fixes it.
export PATH="$SCINTILLA_HOME/.venv-airflow/bin:$PATH"

AIRFLOW_HOME="$SCINTILLA_HOME/airflow"

# The application database. The DAG passes this to every management command it
# runs, so the scheduler and a human running the same command by hand are
# talking to exactly the same place.
DATABASE_URL="postgres://$(whoami)@localhost:5432/scintilla"

# Airflow's own metadata database - a different database on the same server.
AIRFLOW__DATABASE__SQL_ALCHEMY_CONN="postgresql+psycopg2://$(whoami)@localhost:5432/scintilla_airflow"

# LocalExecutor runs tasks as subprocesses of the scheduler. SequentialExecutor
# (the default when the metadata database is SQLite) runs exactly one task at a
# time, which would silently serialise the per-category harvest and hide any
# concurrency bug until deployment.
AIRFLOW__CORE__EXECUTOR="LocalExecutor"

AIRFLOW__CORE__DAGS_FOLDER="$SCINTILLA_HOME/dags"

# The example DAGs are noise in the UI and make it harder to see whether the
# one DAG that matters actually parsed.
AIRFLOW__CORE__LOAD_EXAMPLES="False"

# Airflow defaults to UTC. Being explicit means a schedule reads the same way
# to someone in Geneva as it does here.
AIRFLOW__CORE__DEFAULT_TIMEZONE="utc"

# Parse the DAG folder every 60s rather than every 5s. There is one DAG file;
# re-parsing it constantly is pure CPU cost on a laptop.
AIRFLOW__DAG_PROCESSOR__REFRESH_INTERVAL="60"

# What the DAG harvests, and how much per category per run. Read by the DAG at
# parse time, so changing these needs a re-parse but not a code change.
ARXIV_CATEGORIES="${ARXIV_CATEGORIES:-hep-ex,hep-th,cs.IR}"
ARXIV_DAILY_LIMIT="${ARXIV_DAILY_LIMIT:-200}"

set +a

echo "AIRFLOW_HOME   = $AIRFLOW_HOME"
echo "DAGS           = $AIRFLOW__CORE__DAGS_FOLDER"
echo "metadata DB    = scintilla_airflow"
echo "application DB = scintilla"
echo
echo "Next:  ./.venv-airflow/bin/airflow db migrate"
echo "Then:  ./.venv-airflow/bin/airflow standalone"
