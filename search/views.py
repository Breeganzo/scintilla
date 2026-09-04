"""The search endpoint.

**POST rather than GET.** A query is user input of unbounded shape, and putting
it in a URL means it lands in access logs, proxy logs and browser history. That
matters less for public arXiv metadata than it would elsewhere, but the habit is
worth keeping, and it avoids URL-length limits on long questions. The cost is
that results are not cacheable by URL, which is acceptable while the corpus
changes daily.

**Retrieval returns arXiv IDs; this view hydrates them.** The retrievers know
nothing about the ``Paper`` model beyond its identifier, which is what lets them
be tested without a database and lets the evaluation harness score IDs directly
without loading rows it will not read. Hydration happens here, in one query, in
the order retrieval chose.
"""

from __future__ import annotations

import logging
import time

from drf_spectacular.utils import extend_schema
from rest_framework import status
from rest_framework.response import Response
from rest_framework.views import APIView

from papers.models import Paper
from search.retrievers import RetrievalResult, get_retriever
from search.serializers import (
    SearchRequestSerializer,
    SearchResponseSerializer,
)

logger = logging.getLogger(__name__)


def hydrate(results: list[RetrievalResult]) -> list[dict]:
    """Attach paper metadata to ranked results, preserving retrieval order.

    One query for all of them, then reordered in Python. Fetching per result
    would be an N+1 the moment ``top_k`` grows, and asking the database to
    preserve an arbitrary ordering requires a ``CASE`` expression that is harder
    to read than sorting a list that is at most 50 long.

    A result whose paper has vanished is dropped rather than rendered blank. It
    means the index is ahead of the database - possible if a paper were deleted
    between indexing and querying - and a hit that cannot be displayed is worse
    than one fewer hit. It is logged because it should never happen quietly.
    """
    if not results:
        return []

    papers = Paper.objects.filter(arxiv_id__in=[result.arxiv_id for result in results])
    by_id = {paper.arxiv_id: paper for paper in papers}

    hydrated: list[dict] = []
    for result in results:
        paper = by_id.get(result.arxiv_id)
        if paper is None:
            logger.warning(
                "Retrieved %s is in the index but not in the database; dropping it",
                result.arxiv_id,
            )
            continue
        hydrated.append(
            {
                "arxiv_id": paper.arxiv_id,
                "rank": result.rank,
                "score": result.score,
                "title": paper.title,
                "abstract": paper.abstract,
                "authors": paper.authors,
                "primary_category": paper.primary_category,
                "published_at": paper.published_at,
                "abs_url": paper.abs_url,
                "debug": result.debug,
            }
        )
    return hydrated


class SearchView(APIView):
    """Hybrid search over the corpus."""

    @extend_schema(
        summary="Search the corpus",
        description=(
            "Retrieve papers by keyword (bm25), meaning (dense), or both fused "
            "with Reciprocal Rank Fusion (hybrid). The mode parameter is what "
            "makes the published ablation a measurement of this endpoint rather "
            "than of a parallel implementation."
        ),
        request=SearchRequestSerializer,
        responses={200: SearchResponseSerializer},
        tags=["search"],
    )
    def post(self, request):
        request_serializer = SearchRequestSerializer(data=request.data)
        request_serializer.is_valid(raise_exception=True)
        params = request_serializer.validated_data

        retriever = get_retriever(params["mode"])

        started = time.perf_counter()
        results = retriever.retrieve(params["query"], top_k=params["top_k"])
        took_ms = (time.perf_counter() - started) * 1000

        # Only hybrid reports fusion detail; the baselines have none to report.
        diagnostics = retriever.diagnostics() if hasattr(retriever, "diagnostics") else {}

        payload = {
            "query": params["query"],
            "mode": params["mode"],
            "top_k": params["top_k"],
            "count": len(results),
            "took_ms": round(took_ms, 2),
            "diagnostics": diagnostics,
            "results": hydrate(results),
        }
        # Serialised on the way out as well as in. It costs a few milliseconds
        # and guarantees the response matches the schema the frontend generates
        # its types from.
        return Response(SearchResponseSerializer(payload).data, status=status.HTTP_200_OK)


__all__ = ["SearchView", "hydrate"]
