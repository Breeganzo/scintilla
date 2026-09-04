"""Export a small corpus from the live database for CI to rebuild.

Run locally, against the real corpus, whenever the golden set changes. The
output is committed, so CI never needs a database full of arXiv.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from django.core.management.base import BaseCommand, CommandError

from evaluation.ci_corpus import FIXTURE_PATH, CorpusPaper, gate_queries
from evaluation.golden import load_golden_set
from papers.models import Paper


class Command(BaseCommand):
    help = "Write evaluation/fixtures/ci_corpus.json from the live corpus."

    def add_arguments(self, parser):
        parser.add_argument("--golden-version", type=int, default=1)
        parser.add_argument(
            "--distractors",
            type=int,
            default=250,
            help="Extra irrelevant papers, so retrieval has something to be wrong about.",
        )
        parser.add_argument("--output", type=Path, default=FIXTURE_PATH)

    def handle(self, *args, **options):
        golden = load_golden_set(version=options["golden_version"])
        queries = gate_queries(golden)
        if not queries:
            raise CommandError("No gated queries in the golden set.")

        needed: set[str] = set()
        for query in queries:
            needed.update(query.relevant_ids)

        relevant = list(Paper.objects.filter(arxiv_id__in=needed))
        missing = needed - {p.arxiv_id for p in relevant}
        if missing:
            raise CommandError(
                f"{len(missing)} labelled papers are not in the corpus: {sorted(missing)[:5]}. "
                "A fixture missing them would bake a permanently low score into the baseline."
            )

        # Ordered by arxiv_id rather than sampled randomly, so regenerating the
        # fixture on the same corpus produces a byte-identical file and a diff
        # of this file means the corpus changed, not the sampler.
        distractors = list(
            Paper.objects.exclude(arxiv_id__in=needed).order_by("arxiv_id")[
                : options["distractors"]
            ]
        )

        papers = sorted(
            (CorpusPaper.from_model(p) for p in relevant + distractors),
            key=lambda p: p.arxiv_id,
        )
        payload = {
            "note": (
                "Rebuilt by CI on every run. Not a measure of search quality - see "
                "evaluation/ci_corpus.py for what this does and does not claim."
            ),
            "golden_set_version": golden.version,
            "gated_queries": [q.id for q in queries],
            "papers": [asdict(p) for p in papers],
        }

        output: Path = options["output"]
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

        self.stdout.write(
            self.style.SUCCESS(
                f"Wrote {len(papers)} papers ({len(relevant)} labelled, "
                f"{len(distractors)} distractors) to {output}"
            )
        )
