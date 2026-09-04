"""Pool retrieval candidates for manual relevance judging.

This is a curation aid for building the golden set, not a measurement. It runs
both baseline retrievers over a list of draft queries, takes the union of what
they return, and prints each candidate with enough metadata to judge relevance
by hand.

**Why pool from both retrievers rather than one.** Judging only what BM25
returns would build a golden set that BM25 is guaranteed to score well on: any
paper it missed could never be marked relevant, so it could never be counted
against it. Pooling the union of a lexical and a dense run is the standard
mitigation and it is what TREC has done since 1992. It does not eliminate the
problem - see the pooling-bias section of the golden set README - but it stops
the evaluation from being circular in the most obvious way.

**Why this runs in-process rather than over HTTP.** ``compare_modes.py`` goes
through the API deliberately, because it demonstrates behaviour a user can
reach. This script is building the ruler, not using it, so speed matters more
than fidelity and there is no throttle to respect.

Usage::

    python scripts/pool_candidates.py drafts.json > pool.txt

where ``drafts.json`` is a list of ``{"id": ..., "query": ..., "class": ...}``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings.dev")

import django  # noqa: E402

django.setup()

from papers.models import Paper  # noqa: E402
from search.retrievers import BM25Retriever, DenseRetriever  # noqa: E402

ABSTRACT_CHARS = 240


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("drafts", type=Path)
    parser.add_argument("--depth", type=int, default=12)
    args = parser.parse_args()

    drafts = json.loads(args.drafts.read_text())
    bm25 = BM25Retriever()
    dense = DenseRetriever()

    for draft in drafts:
        lexical = bm25.retrieve(draft["query"], top_k=args.depth)
        semantic = dense.retrieve(draft["query"], top_k=args.depth)

        lex_rank = {r.arxiv_id: r.rank for r in lexical}
        sem_rank = {r.arxiv_id: r.rank for r in semantic}
        # Sort by best rank in either run so the most plausible candidates are
        # judged first, while still showing everything in the pool.
        pool = sorted(
            set(lex_rank) | set(sem_rank),
            key=lambda pid: min(lex_rank.get(pid, 999), sem_rank.get(pid, 999)),
        )
        papers = {p.arxiv_id: p for p in Paper.objects.filter(arxiv_id__in=pool)}

        print("=" * 100)
        print(f"[{draft['id']}] ({draft['class']}) {draft['query']}")
        print("-" * 100)
        for pid in pool:
            paper = papers.get(pid)
            if paper is None:
                continue
            marks = f"b{lex_rank.get(pid, '-'):>3} d{sem_rank.get(pid, '-'):>3}"
            abstract = " ".join(paper.abstract.split())[:ABSTRACT_CHARS]
            print(f"{marks} | {pid} | {paper.primary_category}")
            print(f"      T: {paper.title}")
            print(f"      A: {abstract}")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
