"""Run the ablation and write the results down.

Every run records the corpus size and the commit that produced it. A metric
without that context cannot be compared to anything: a later run against a
larger corpus is not an improvement or a regression, it is a different
experiment, and the only way to notice is to have stored the provenance at the
time.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from django.conf import settings
from django.core.management.base import BaseCommand, CommandError
from django.db import transaction

from evaluation.harness import run_ablation
from evaluation.models import EvaluationRun, QueryResult
from evaluation.report import to_json, to_markdown
from papers.models import Chunk, Paper
from search.retrievers import DEFAULT_TOP_K, MODES

DEFAULT_OUTPUT_DIR = Path(settings.BASE_DIR) / "evaluation" / "results"


def current_git_sha() -> str:
    """The commit under test, or empty if git cannot answer.

    Failing softly here is deliberate: an unversioned checkout should still be
    able to produce numbers for a human reading them interactively. Day 10's
    regression gate runs in CI, where the SHA is always available.
    """
    try:
        result = subprocess.run(  # noqa: S603
            ["git", "rev-parse", "HEAD"],  # noqa: S607
            cwd=settings.BASE_DIR,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    return result.stdout.strip() if result.returncode == 0 else ""


class Command(BaseCommand):
    help = "Sweep every retrieval mode over the golden set and write JSON + markdown."

    def add_arguments(self, parser):
        parser.add_argument("--golden-version", type=int, default=1)
        parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
        parser.add_argument("--modes", nargs="+", default=list(MODES))
        parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
        parser.add_argument(
            "--no-save",
            action="store_true",
            help="Skip the database write. Files are still produced.",
        )
        parser.add_argument(
            "--no-warmup",
            action="store_true",
            help="Time the first query cold. Only useful for measuring model load.",
        )

    def handle(self, *args, **options):
        from evaluation.golden import load_golden_set

        golden = load_golden_set(version=options["golden_version"])

        corpus_papers = Paper.objects.count()
        corpus_chunks = Chunk.objects.count()
        if not corpus_papers:
            raise CommandError(
                "The corpus is empty. Run `manage.py ingest_arxiv` and "
                "`manage.py index_papers` before evaluating."
            )

        # Cheap guard against the most damaging silent failure: labels that do
        # not resolve score zero for every mode, which reads as a hard query
        # rather than a broken one.
        missing = golden.relevant_ids - set(
            Paper.objects.filter(arxiv_id__in=golden.relevant_ids).values_list(
                "arxiv_id", flat=True
            )
        )
        if missing:
            raise CommandError(
                f"{len(missing)} labelled papers are not in this corpus, so every mode "
                f"would be scored against unreachable answers. Run validate_golden_set. "
                f"First few: {sorted(missing)[:5]}"
            )

        self.stdout.write(
            f"golden v{golden.version}: {len(golden)} queries "
            f"({len(golden.by_class('unanswerable'))} unanswerable) "
            f"against {corpus_papers} papers / {corpus_chunks} chunks"
        )

        reports = run_ablation(
            modes=options["modes"],
            golden=golden,
            top_k=options["top_k"],
            warm_up=not options["no_warmup"],
        )

        git_sha = current_git_sha()
        meta = {
            "golden_set_version": golden.version,
            "top_k": options["top_k"],
            "corpus_papers": corpus_papers,
            "corpus_chunks": corpus_chunks,
            "git_sha": git_sha,
        }

        output_dir: Path = options["output_dir"]
        output_dir.mkdir(parents=True, exist_ok=True)
        json_path = output_dir / "ablation.json"
        markdown_path = output_dir / "ablation.md"

        json_path.write_text(to_json(reports, meta) + "\n")
        markdown_path.write_text(to_markdown(reports, meta) + "\n")

        if not options["no_save"]:
            self._persist(reports, meta)

        self.stdout.write("")
        self.stdout.write(to_markdown(reports, meta))
        self.stdout.write(self.style.SUCCESS(f"wrote {json_path}"))
        self.stdout.write(self.style.SUCCESS(f"wrote {markdown_path}"))

    @transaction.atomic
    def _persist(self, reports, meta) -> None:
        """One row per mode, plus every query, in a single transaction.

        Atomic because a half-written run is worse than no run: the aggregate
        row would exist for the regression check to read while the per-query
        detail needed to explain it would not.
        """
        for mode, report in reports.items():
            run = EvaluationRun.objects.create(
                golden_set_version=report.golden_set_version,
                mode=mode,
                top_k=report.top_k,
                corpus_papers=meta["corpus_papers"],
                corpus_chunks=meta["corpus_chunks"],
                git_sha=meta["git_sha"],
                metrics={
                    "overall": report.overall,
                    "by_class": report.by_class(),
                    "median_latency_ms": report.median_latency_ms,
                },
            )
            QueryResult.objects.bulk_create(
                QueryResult(
                    run=run,
                    query_id=outcome.query_id,
                    query_class=outcome.query_class,
                    retrieved_ids=outcome.retrieved_ids,
                    relevant_ids=outcome.relevant_ids,
                    metrics=outcome.metrics,
                )
                for outcome in report.outcomes
            )
            self.stdout.write(f"saved run {run.pk}: {json.dumps(report.overall, sort_keys=True)}")
