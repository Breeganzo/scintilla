"""A small, self-contained corpus that CI can rebuild from scratch.

**Why CI cannot just measure the real corpus.** The published ablation runs
against 3,377 papers held in a local Postgres and a local OpenSearch. A GitHub
runner has neither, and shipping 3,377 abstracts plus their embeddings into the
repository to fake it would make every clone heavier and still not exercise the
code that produced them.

**What this replaces it with.** A few hundred abstracts, stored as raw text, and
rebuilt on every CI run: chunked by the real chunker, embedded by the
deterministic test embedder, written through the real vector store and searched
by the real retriever. Nothing about the pipeline is stubbed except the model
weights, so a change that breaks chunking, vector storage or ranking changes the
score.

**What it deliberately does not claim.** These numbers are not a measure of
search quality and are not comparable to the ablation. The embedder is a hashing
bag-of-words, not BGE, so it has lexical signal and no semantic signal; the
corpus is small, so recall is inflated by having fewer papers to be wrong about.
The only question this fixture answers is "did today's commit retrieve worse
than yesterday's, on identical inputs?" - and for that, a cheap deterministic
setup is better than an expensive realistic one, because a gate that takes ten
minutes and occasionally flakes is a gate that gets disabled.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from django.conf import settings
from django.db import transaction
from django.utils import timezone

from evaluation.golden import GoldenSet, load_golden_set
from evaluation.metrics import score_query
from ingestion.chunking import chunk_paper
from ingestion.embedding import HashingEmbedder
from ingestion.vectors import store_vectors
from papers.models import Chunk, Paper
from search.retrievers.base import DEFAULT_TOP_K, collapse_to_papers

FIXTURE_PATH = Path(settings.BASE_DIR) / "evaluation" / "fixtures" / "ci_corpus.json"

# The classes the hashing embedder can actually rank. Paraphrase and conceptual
# queries share almost no vocabulary with their relevant papers - that is what
# makes them paraphrase and conceptual - so a bag-of-words embedder scores near
# zero on them, and a metric pinned near zero cannot detect a regression. They
# are excluded from the gate rather than reported as failures.
GATE_CLASSES = ("exact_term", "multi_hop")

# Metrics the gate watches. recall@1 is excluded: on forty-odd queries it moves
# in steps of about 0.02 and is the noisiest thing in the table.
GATE_METRICS = ("recall@5", "recall@10", "mrr", "ndcg@10")


@dataclass(frozen=True)
class CorpusPaper:
    """The minimum needed to rebuild a paper. No chunks, no vectors."""

    arxiv_id: str
    title: str
    abstract: str
    primary_category: str
    categories: list[str]

    @classmethod
    def from_model(cls, paper: Paper) -> CorpusPaper:
        return cls(
            arxiv_id=paper.arxiv_id,
            title=paper.title,
            abstract=paper.abstract,
            primary_category=paper.primary_category,
            categories=list(paper.categories),
        )


def load_fixture(path: Path = FIXTURE_PATH) -> list[CorpusPaper]:
    payload = json.loads(path.read_text())
    return [CorpusPaper(**record) for record in payload["papers"]]


@transaction.atomic
def seed_corpus(papers: list[CorpusPaper]) -> int:
    """Rebuild the corpus in the current database, chunking and embedding it.

    Runs the real :func:`chunk_paper` and the real :func:`store_vectors`, which
    is the whole point: a fixture of pre-computed embeddings would freeze the
    pipeline's output and the gate would keep passing after the pipeline broke.
    """
    embedder = HashingEmbedder(dimension=settings.EMBEDDING_DIM)
    now = timezone.now()
    created = 0

    for record in papers:
        paper = Paper.objects.create(
            arxiv_id=record.arxiv_id,
            version=1,
            title=record.title,
            abstract=record.abstract,
            authors=[],
            categories=record.categories,
            primary_category=record.primary_category,
            published_at=now,
            arxiv_updated_at=now,
            abs_url=f"https://arxiv.org/abs/{record.arxiv_id}",
            pdf_url=f"https://arxiv.org/pdf/{record.arxiv_id}",
        )
        pieces = chunk_paper(
            record.title,
            record.abstract,
            arxiv_id=record.arxiv_id,
            count_tokens=embedder.count_tokens,
        )
        chunks = Chunk.objects.bulk_create(
            [
                Chunk(
                    paper=paper,
                    chunk_index=piece.index,
                    text=piece.text,
                    token_count=piece.token_count,
                    embedding_model=embedder.model_name,
                    embedding_dim=embedder.dimension,
                )
                for piece in pieces
            ]
        )
        store_vectors(chunks, embedder.embed_documents([c.text for c in chunks]))
        created += len(chunks)

    return created


def gate_queries(golden: GoldenSet) -> list:
    return [q for q in golden.queries if q.query_class in GATE_CLASSES]


def score_corpus(
    golden: GoldenSet | None = None,
    top_k: int = DEFAULT_TOP_K,
) -> dict[str, Any]:
    """Retrieve every gated query against the seeded corpus and score it.

    Deliberately does not go through ``get_retriever("dense")``: that resolves
    the configured production embedder, which would download BGE. The retrieval
    *path* below - nearest neighbour, collapse to papers, rank - is the same
    code, exercised with the deterministic embedder substituted at the one
    point where realism costs more than it buys.
    """
    from ingestion.vectors import knn_search

    golden = golden or load_golden_set()
    embedder = HashingEmbedder(dimension=settings.EMBEDDING_DIM)
    queries = gate_queries(golden)

    per_query: dict[str, dict[str, float]] = {}
    for query in queries:
        # Over-fetch before collapsing, for the same reason the real dense
        # retriever does: several chunks of one paper can occupy the top slots,
        # and cutting at top_k first would return fewer papers than asked for.
        rows = knn_search(embedder.embed_query(query.query), size=top_k * 5)
        results = collapse_to_papers([(c.paper.arxiv_id, score, {}) for c, score in rows])[:top_k]
        per_query[query.id] = score_query(
            [r.arxiv_id for r in results], list(query.relevant_ids), ks=(1, 5, 10)
        )

    overall = {
        metric: sum(scores[metric] for scores in per_query.values()) / len(per_query)
        for metric in GATE_METRICS
    }
    return {
        "golden_set_version": golden.version,
        "queries": len(queries),
        "papers": Paper.objects.count(),
        "chunks": Chunk.objects.count(),
        "top_k": top_k,
        "overall": overall,
        "per_query": per_query,
    }


__all__ = [
    "FIXTURE_PATH",
    "GATE_CLASSES",
    "GATE_METRICS",
    "CorpusPaper",
    "gate_queries",
    "load_fixture",
    "score_corpus",
    "seed_corpus",
]
