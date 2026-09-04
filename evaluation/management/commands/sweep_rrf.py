"""Measure RRF's ``k`` instead of asserting it.

``search/fusion.py`` adopts ``k = 60`` from Cormack et al. (SIGIR 2009) and says
plainly that it is a convention, exposed as a parameter because it should be
measured rather than believed. This is that measurement.

**What ``k`` controls.** It is the constant added to the rank before taking the
reciprocal, so it sets how sharply the top of each list dominates. At ``k = 0``
rank 1 contributes 1.0 and rank 2 contributes 0.5, so one retriever's confident
top hit can carry the fused list. At ``k = 60`` those contributions are 0.0164
and 0.0161 - almost identical - so the ordering is decided by *agreement between
the lists* rather than by either list's conviction.

That distinction is the whole diagnosis of why hybrid lost. If hybrid improves
as ``k`` falls, the fused list was over-rewarding agreement between a strong
retriever and a weak one. If it does not, the problem is upstream of fusion.

**These are not shipped modes.** Every row is labelled ``hybrid@k=N`` so a
sweep result cannot be quoted as though the API serves it.
"""

from __future__ import annotations

from django.core.management.base import BaseCommand, CommandError

from evaluation.golden import load_golden_set
from evaluation.harness import run_mode
from evaluation.report import HEADLINE, headline_table, per_class_table
from papers.models import Paper
from search.retrievers import DEFAULT_TOP_K
from search.retrievers.hybrid import HybridRetriever

# Spans the two regimes: 0 lets a single confident retriever dominate, 60 is the
# published default, 120 pushes even harder towards requiring agreement.
DEFAULT_KS = (0, 1, 5, 10, 20, 60, 120)


class Command(BaseCommand):
    help = "Score hybrid retrieval at several RRF k values against the golden set."

    def add_arguments(self, parser):
        parser.add_argument("--golden-version", type=int, default=1)
        parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
        parser.add_argument("--rrf-k", nargs="+", type=int, default=list(DEFAULT_KS))

    def handle(self, *args, **options):
        if not Paper.objects.exists():
            raise CommandError("The corpus is empty; there is nothing to retrieve from.")

        golden = load_golden_set(version=options["golden_version"])
        reports = {}

        # The baselines belong in the same table. A sweep that only shows hybrid
        # variants can demonstrate a best k without revealing that every one of
        # them is behind a plain dense search.
        for mode in ("bm25", "dense"):
            reports[mode] = run_mode(mode, golden=golden, top_k=options["top_k"])
            self.stdout.write(f"scored {mode}")

        for k in options["rrf_k"]:
            label = f"hybrid@k={k}"
            reports[label] = run_mode(
                "hybrid",
                golden=golden,
                top_k=options["top_k"],
                retriever=HybridRetriever(k=k),
                label=label,
                warm_up=False,
            )
            self.stdout.write(f"scored {label}")

        self.stdout.write("")
        self.stdout.write(headline_table(reports))
        self.stdout.write("")
        self.stdout.write("recall@10 by class")
        self.stdout.write(per_class_table(reports, "recall@10"))

        best = max(reports, key=lambda name: reports[name].overall.get("ndcg@10", 0.0))
        self.stdout.write("")
        self.stdout.write(f"best nDCG@10: {best}")
        for metric in HEADLINE:
            leader = max(reports, key=lambda name: reports[name].overall.get(metric, 0.0))
            self.stdout.write(f"  {metric}: {leader}")
