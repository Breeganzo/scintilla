"""The regression gate: rebuild the CI corpus, score it, compare to the baseline.

**What a gate has to be to be worth having.** It has to fail when the code gets
worse and pass when it does not, and it has to be believable enough that nobody
deletes it. Those pull in opposite directions: a tight tolerance catches more
regressions and also fires on noise, and a gate that cries wolf gets an
``if: false`` added to it within a fortnight.

The tolerance here is 2 percentage points, absolute, per metric. That is chosen
against the observed behaviour of this particular measurement rather than
picked because it is a round number: the corpus, the embedder and the query set
are all fixed, and the pipeline is deterministic, so two runs of unchanged code
produce *identical* scores. The tolerance is not absorbing run-to-run noise -
there is none - it is absorbing the small, legitimate movements that come from
changing the chunker or the fixture on purpose. Anything larger than that is a
decision someone should have to make deliberately, by updating the baseline in
a commit that says why.

**Why it compares per-metric and not an average.** A single blended score lets a
collapse in one metric hide behind an improvement in another. Every gated metric
must hold on its own.
"""

from __future__ import annotations

import json
from pathlib import Path

from django.conf import settings
from django.core.management.base import BaseCommand, CommandError

from evaluation.ci_corpus import GATE_METRICS, load_fixture, score_corpus, seed_corpus
from evaluation.golden import load_golden_set
from papers.models import Paper

BASELINE_PATH = Path(settings.BASE_DIR) / "evaluation" / "results" / "ci_baseline.json"

# Absolute, in metric units. 0.02 on recall@10 over ~20 gated queries is roughly
# one query moving one position across the cutoff.
DEFAULT_TOLERANCE = 0.02


def compare(
    baseline: dict, current: dict, tolerance: float = DEFAULT_TOLERANCE
) -> tuple[list[str], list[str]]:
    """Return ``(failures, notes)``.

    Provenance is checked before metrics. A run against a different corpus or a
    different golden set is a different experiment, not a regression, and
    reporting it as one teaches people to ignore the gate.
    """
    failures: list[str] = []
    notes: list[str] = []

    for key in ("papers", "chunks", "golden_set_version", "queries", "top_k"):
        if baseline.get(key) != current.get(key):
            failures.append(
                f"{key} changed: baseline {baseline.get(key)}, now {current.get(key)}. "
                "The comparison is not like-for-like; regenerate the baseline if this is intended."
            )

    for metric in GATE_METRICS:
        before = baseline["overall"].get(metric)
        after = current["overall"].get(metric)
        if before is None or after is None:
            failures.append(f"{metric} is missing from one of the two runs.")
            continue
        delta = after - before
        line = f"{metric}: {before:.4f} -> {after:.4f} ({delta:+.4f})"
        if delta < -tolerance:
            failures.append(f"{line} - dropped more than the {tolerance:.2f} tolerance.")
        else:
            notes.append(line)

    return failures, notes


def regressed_queries(baseline: dict, current: dict, metric: str = "ndcg@10") -> list[str]:
    """Queries that got worse, for the message. A rate says nothing actionable."""
    before = baseline.get("per_query", {})
    after = current.get("per_query", {})
    worse = [
        f"  {qid}: {before[qid][metric]:.3f} -> {after[qid][metric]:.3f}"
        for qid in sorted(before)
        if qid in after and after[qid].get(metric, 0) < before[qid].get(metric, 0) - 1e-9
    ]
    return worse


class Command(BaseCommand):
    help = "Fail if retrieval got worse than the committed baseline."

    def add_arguments(self, parser):
        parser.add_argument("--baseline", type=Path, default=BASELINE_PATH)
        parser.add_argument("--tolerance", type=float, default=DEFAULT_TOLERANCE)
        parser.add_argument(
            "--write-baseline",
            action="store_true",
            help="Record the current scores as the new baseline instead of checking them.",
        )

    def handle(self, *args, **options):
        # The gate rebuilds a corpus from scratch, so it must never be pointed
        # at a database that already holds one. Refusing is not paranoia: the
        # command is meant to be run in CI against an empty database, and the
        # obvious local mistake is to run it against the development database,
        # where it would both collide on arxiv_id and quietly measure a corpus
        # 3,000 papers larger than the baseline's.
        existing = Paper.objects.count()
        if existing:
            raise CommandError(
                f"The database already contains {existing} papers. This command rebuilds "
                "the corpus from a fixture and must be run against an empty database."
            )

        papers = load_fixture()
        self.stdout.write(f"Seeding {len(papers)} papers...")
        chunks = seed_corpus(papers)
        self.stdout.write(f"Indexed {chunks} chunks. Scoring...")

        current = score_corpus(golden=load_golden_set())
        baseline_path: Path = options["baseline"]

        if options["write_baseline"]:
            baseline_path.parent.mkdir(parents=True, exist_ok=True)
            baseline_path.write_text(json.dumps(current, indent=2, sort_keys=True) + "\n")
            self.stdout.write(self.style.SUCCESS(f"Wrote baseline to {baseline_path}"))
            for metric in GATE_METRICS:
                self.stdout.write(f"  {metric}: {current['overall'][metric]:.4f}")
            return

        if not baseline_path.exists():
            raise CommandError(
                f"No baseline at {baseline_path}. Run with --write-baseline to create one."
            )

        baseline = json.loads(baseline_path.read_text())
        failures, notes = compare(baseline, current, tolerance=options["tolerance"])

        for line in notes:
            self.stdout.write(f"  {line}")

        if not failures:
            self.stdout.write(self.style.SUCCESS("No regression."))
            return

        for line in failures:
            self.stderr.write(self.style.ERROR(f"  {line}"))
        worse = regressed_queries(baseline, current)
        if worse:
            self.stderr.write("\nQueries that got worse (nDCG@10):")
            for line in worse:
                self.stderr.write(line)
        raise CommandError(f"{len(failures)} regression check(s) failed.")
