"""Scheduled arXiv harvesting and indexing.

The DAG is deliberately thin. It shells out to the same management commands a
human would run - ``ingest_arxiv`` and ``index_papers`` - rather than importing
Django and calling the pipeline directly. Two reasons:

* **Dependency isolation.** Airflow pins a large tree of its own. Running it in
  the same interpreter as Django means neither can be upgraded without
  negotiating with the other. Here the orchestrator and the application share
  nothing but a database and a process boundary.
* **One interface, not two.** If the DAG called ``ingestion.pipeline.ingest()``
  directly it would be a second entry point that could drift from the command
  people actually run and test. Shelling out means the scheduled path and the
  manual path are the same path, and a bug in one is a bug in both.

The cost is that failures arrive as a non-zero exit code and a log, not a Python
traceback in the task context. That is an acceptable trade: both commands
already raise ``CommandError`` on failure, which Django turns into exit status 1,
which Airflow turns into a failed task.

Task boundaries
---------------
Tasks are split where it is *safe to retry*, not at every logical step of the
pipeline. Splitting "fetch / parse / upsert / chunk" into separate tasks would
mean moving thousands of papers between them through XCom, which is a metadata
database, not a data bus. Instead:

* one harvest task **per category**, so a failure in ``cs.IR`` neither blocks
  nor re-runs ``hep-ex``, and a retry re-fetches only what failed;
* one indexing task, because embedding and indexing are already incremental -
  it picks up whatever is unindexed regardless of which harvest produced it;
* one verification task, which fails the run if the stores disagree.

Every task is idempotent, which is what makes ``retries`` safe. Re-running the
whole DAG on an unchanged corpus does nothing at all - that is the Phase 2 gate.
"""

from __future__ import annotations

import os
from datetime import timedelta
from pathlib import Path

import pendulum
from airflow.providers.standard.operators.bash import BashOperator
from airflow.sdk import dag, task

# The DAG file lives in <project>/dags/, so the project root is one level up.
# Deriving it rather than hard-coding a path means the DAG works from a clone in
# any directory, which is the same property the fresh-clone test checks for.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PYTHON_BIN = PROJECT_ROOT / ".venv" / "bin" / "python"
MANAGE = PROJECT_ROOT / "manage.py"

# Categories are read from the environment so that changing what gets harvested
# does not require editing - and redeploying - the DAG.
CATEGORIES = [
    category.strip()
    for category in os.environ.get("ARXIV_CATEGORIES", "hep-ex,hep-th,cs.IR").split(",")
    if category.strip()
]

# Deliberately modest. arXiv asks for one request every three seconds and this
# runs daily; the watermark means a steady-state run fetches only what is new.
PAPERS_PER_CATEGORY = int(os.environ.get("ARXIV_DAILY_LIMIT", "200"))

# Passed through to every command. The scheduler's environment is not a login
# shell, so PATH has to be set explicitly or psql's libraries are not found.
COMMAND_ENV = {
    "DATABASE_URL": os.environ.get("DATABASE_URL", ""),
    "OPENSEARCH_URL": os.environ.get("OPENSEARCH_URL", "http://localhost:9200"),
    "PATH": os.environ.get("PATH", ""),
    "HOME": os.environ.get("HOME", ""),
    "DJANGO_SETTINGS_MODULE": os.environ.get("DJANGO_SETTINGS_MODULE", "config.settings.dev"),
}


@dag(
    dag_id="arxiv_ingest",
    description="Harvest arXiv, embed into pgvector, index into OpenSearch",
    # 06:00 UTC. arXiv publishes new listings overnight; running after that
    # means a daily run usually has something to do.
    schedule="0 6 * * *",
    start_date=pendulum.datetime(2026, 9, 1, tz="UTC"),
    # Without this, enabling a DAG with a start_date in the past immediately
    # queues one run per missed interval. Backfilling arXiv by date is
    # meaningless here anyway: the watermark already tracks how far each
    # category has been harvested, so a single run catches up on its own.
    catchup=False,
    # Two concurrent runs would race on the same chunks - both would see the
    # same unindexed rows, embed them twice and write the same documents twice.
    # The writes are idempotent so the result would still be correct, but the
    # work would be wasted and the run ledger would be misleading.
    max_active_runs=1,
    default_args={
        "retries": 2,
        "retry_delay": timedelta(minutes=5),
        "retry_exponential_backoff": True,
        "max_retry_delay": timedelta(minutes=30),
        # A harvest that has not finished in an hour is stuck, not slow.
        "execution_timeout": timedelta(hours=1),
    },
    tags=["scintilla", "ingestion"],
    doc_md=__doc__,
)
def arxiv_ingest() -> None:
    @task
    def preflight() -> dict[str, str]:
        """Fail early and legibly if a dependency is missing.

        Without this, an unreachable OpenSearch surfaces halfway through the
        run as a connection error inside a bulk request, after the harvest has
        already spent several minutes on the arXiv API. Checking first turns a
        confusing partial failure into an obvious one.
        """
        import urllib.parse
        import urllib.request

        problems: list[str] = []

        if not PYTHON_BIN.exists():
            problems.append(f"application virtualenv missing at {PYTHON_BIN}")
        if not MANAGE.exists():
            problems.append(f"manage.py missing at {MANAGE}")
        if not COMMAND_ENV["DATABASE_URL"]:
            problems.append("DATABASE_URL is not set in the scheduler environment")

        opensearch_url = COMMAND_ENV["OPENSEARCH_URL"]
        # urlopen will happily accept file:// or a custom scheme. This URL comes
        # from the environment, so check it is HTTP before dereferencing it -
        # otherwise a typo in .env turns a health check into a file read.
        if urllib.parse.urlparse(opensearch_url).scheme not in {"http", "https"}:
            problems.append(f"OPENSEARCH_URL must be http or https, got {opensearch_url!r}")
        else:
            try:
                with urllib.request.urlopen(  # noqa: S310 - scheme validated above
                    f"{opensearch_url}/_cluster/health", timeout=10
                ) as r:
                    if r.status != 200:
                        problems.append(f"OpenSearch returned HTTP {r.status}")
            except (TimeoutError, OSError) as exc:
                problems.append(f"OpenSearch unreachable at {opensearch_url}: {exc}")

        if problems:
            raise RuntimeError("Preflight failed:\n  - " + "\n  - ".join(problems))

        return {"categories": ",".join(CATEGORIES), "opensearch": opensearch_url}

    harvest = BashOperator.partial(
        task_id="harvest",
        # `set -euo pipefail` because a bare bash task returns the exit status
        # of the *last* command; without it a failure early in a chain is
        # invisible and the task is reported as successful.
        bash_command=(
            "set -euo pipefail\n"
            f"cd {PROJECT_ROOT}\n"
            f"{PYTHON_BIN} {MANAGE} ingest_arxiv "
            "--categories '{{ params.category }}' "
            f"--limit {PAPERS_PER_CATEGORY} "
            "--page-size 100 "
            "--triggered-by scheduler"
        ),
        env=COMMAND_ENV,
        # The harvest talks to a third party, so a failure is usually transient.
        retries=3,
    ).expand(params=[{"category": category} for category in CATEGORIES])

    index = BashOperator(
        task_id="index",
        bash_command=(f"set -euo pipefail\ncd {PROJECT_ROOT}\n{PYTHON_BIN} {MANAGE} index_papers"),
        env=COMMAND_ENV,
        # Embedding the backlog on a cold start is slower than a steady-state
        # run, which only touches what changed.
        execution_timeout=timedelta(hours=2),
    )

    @task
    def verify() -> dict[str, int]:
        """Assert the two stores agree, and fail the run if they do not.

        A pipeline that reports success while the dense and lexical indexes hold
        different documents is worse than one that fails: the gap only shows up
        later as a retrieval result that is quietly missing a paper. This is the
        cheap version of the invariant Phase 3 will check properly.
        """
        import json
        import subprocess

        script = (
            "import json, django, os;"
            "os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings.dev');"
            "django.setup();"
            "from ingestion import vectors;"
            "from ingestion.indexing import SearchIndex, build_client;"
            "from papers.models import Paper, Chunk;"
            "print(json.dumps({"
            "'papers': Paper.objects.count(),"
            "'chunks': Chunk.objects.count(),"
            "'indexed_papers': Paper.objects.filter(indexed_at__isnull=False).count(),"
            "'unembedded': vectors.unembedded_count(),"
            "'documents': SearchIndex(build_client()).count(),"
            "}))"
        )
        # Both the interpreter path and the script are module-level literals -
        # no part of this command comes from a request, a DAG param or a file.
        completed = subprocess.run(  # noqa: S603 - fixed interpreter, literal script
            [str(PYTHON_BIN), "-c", script],
            cwd=PROJECT_ROOT,
            env={**os.environ, **COMMAND_ENV},
            capture_output=True,
            text=True,
            check=True,
        )
        counts = json.loads(completed.stdout.strip().splitlines()[-1])

        problems: list[str] = []
        if counts["unembedded"]:
            problems.append(f"{counts['unembedded']} chunks have no embedding")
        if counts["documents"] != counts["chunks"]:
            problems.append(
                f"OpenSearch holds {counts['documents']} documents "
                f"but PostgreSQL holds {counts['chunks']} chunks"
            )
        if counts["indexed_papers"] != counts["papers"]:
            problems.append(
                f"{counts['papers'] - counts['indexed_papers']} papers are not marked indexed"
            )
        if problems:
            raise RuntimeError("Verification failed:\n  - " + "\n  - ".join(problems))

        return counts

    preflight() >> harvest >> index >> verify()


arxiv_ingest()
