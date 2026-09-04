"""The OpenSearch index: mapping, bulk indexing, and rebuilds behind an alias.

**This index is lexical only.** BM25 over analysed text lives here; the dense
vectors live in PostgreSQL via pgvector, for the reasons set out in
``ingestion.vectors``. The short version is that OpenSearch publishes no macOS
build and its k-NN plugin has no darwin native libraries, so keeping vectors
here would have meant a development environment unable to run the system being
developed.

Both halves are still written in one pass by :func:`index_corpus`, and a chunk
is marked as indexed only once both have succeeded. A document that is in
OpenSearch but has no vector is retrievable by BM25 and invisible to dense
search, which would surface in Phase 3 as an unexplained gap between the two
retrievers rather than as an error.

**The mapping is the part that is painful to change later.** A field's type is
fixed at index creation; changing it means reindexing everything. Two choices in
``build_index_body`` are deliberate:

* ``dynamic: "strict"``. A field that is not in the mapping raises instead of
  being silently invented. The default would let a typo like ``publised_at``
  create a real field, index nothing useful into the intended one, and produce a
  query returning zero hits for reasons no error explains.
* The ``english`` analyzer on every text field. Stemming and stopword removal
  are what make "measuring the decay" match "measurement of decays"; without it
  BM25 matches only exact surface forms.

**Document identity is deterministic**: ``arxiv_id#chunk_index``. This is the
third idempotency mechanism from ``pipeline``. Re-indexing a paper overwrites
its documents rather than appending new ones, so the index cannot accumulate
duplicates no matter how many times ingestion is re-run.

**Rebuilds go through an alias.** The application always queries the alias,
never a concrete index. A rebuild writes a new timestamped index and then moves
the alias in a single atomic call, so the change is invisible to readers and the
previous index is still there if the new one is wrong.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from django.conf import settings
from django.db.models import Q
from django.utils import timezone
from opensearchpy import OpenSearch
from opensearchpy.exceptions import NotFoundError

from ingestion.embedding import Embedder, get_embedder
from ingestion.vectors import store_vectors
from papers.models import Chunk, Paper

logger = logging.getLogger(__name__)

# The bulk API takes a whole batch in one request. 500 is a common working
# default: large enough that per-request overhead stops mattering, small enough
# that one oversized request does not exceed the default 100 MB HTTP limit.
DEFAULT_BULK_SIZE = 500

# hep-ex collaboration papers carry author lists in the thousands. The full
# list stays in PostgreSQL, which is what the API serves; the index keeps only
# enough for faceting. Indexing 3,000 keyword terms per document would inflate
# the index for a feature nobody uses past the first few names.
MAX_INDEXED_AUTHORS = 20


class IndexingError(RuntimeError):
    """Raised when the index cannot be created or written to."""


@dataclass
class IndexingResult:
    """What one indexing pass did."""

    index: str
    documents_seen: int = 0
    documents_indexed: int = 0
    documents_failed: int = 0
    papers_marked: int = 0
    failures: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.documents_failed == 0


def build_index_body(
    *,
    shards: int = 1,
    replicas: int = 0,
) -> dict[str, Any]:
    """The index settings and mapping.

    One shard and zero replicas because this is a single node. A replica on a
    single-node cluster can never be allocated, which leaves the cluster health
    permanently yellow and trains you to ignore the one signal that would tell
    you something is actually wrong.
    """
    return {
        "settings": {
            "index": {
                "number_of_shards": shards,
                "number_of_replicas": replicas,
            }
        },
        "mappings": {
            "dynamic": "strict",
            "properties": {
                "arxiv_id": {"type": "keyword"},
                "chunk_index": {"type": "integer"},
                "version": {"type": "integer"},
                # The english analyzer stems and strips stopwords, so
                # "measuring the decay" matches "measurement of decays".
                # Without it BM25 matches only exact surface forms.
                "title": {"type": "text", "analyzer": "english"},
                "abstract": {"type": "text", "analyzer": "english"},
                # Exactly the text that was embedded, kept so a retrieved hit
                # can be shown as the model saw it rather than reassembled.
                "text": {"type": "text", "analyzer": "english"},
                "authors": {"type": "keyword"},
                "categories": {"type": "keyword"},
                "primary_category": {"type": "keyword"},
                "published_at": {"type": "date"},
            },
        },
    }


def document_id(arxiv_id: str, chunk_index: int) -> str:
    """The deterministic document ID.

    Includes the chunk index even though there is one chunk per paper today,
    so that splitting a paper later adds documents instead of silently
    overwriting chunk 0 with chunk 1.
    """
    return f"{arxiv_id}#{chunk_index}"


def build_document(chunk: Chunk) -> dict[str, Any]:
    """One indexable document from a chunk.

    No vector: the embedding lives on the ``Chunk`` row in PostgreSQL.
    """
    paper = chunk.paper
    return {
        "arxiv_id": paper.arxiv_id,
        "chunk_index": chunk.chunk_index,
        "version": paper.version,
        "title": paper.title,
        "abstract": paper.abstract,
        "text": chunk.text,
        "authors": paper.authors[:MAX_INDEXED_AUTHORS],
        "categories": paper.categories,
        "primary_category": paper.primary_category,
        "published_at": paper.published_at.isoformat(),
    }


def build_client(url: str | None = None) -> OpenSearch:
    """A configured OpenSearch client."""
    return OpenSearch(
        hosts=[url or settings.OPENSEARCH_URL],
        # Fail rather than hang. An indexer blocked forever on an unresponsive
        # node looks identical to one doing slow work.
        timeout=30,
        max_retries=3,
        retry_on_timeout=True,
    )


class SearchIndex:
    """Everything this project does to an OpenSearch index."""

    def __init__(self, client: OpenSearch, alias: str | None = None) -> None:
        self.client = client
        self.alias = alias or settings.OPENSEARCH_INDEX

    # -- lifecycle ---------------------------------------------------------

    def versioned_name(self, moment: datetime | None = None) -> str:
        """A unique name for a new index.

        The counter matters: two rebuilds in the same second would otherwise
        produce the same name, and the second would silently write into the
        first's index instead of building a clean one.
        """
        moment = moment or datetime.now(UTC)
        base = f"{self.alias}-{moment:%Y%m%d%H%M%S}"
        name, suffix = base, 1
        while self.client.indices.exists(index=name):
            suffix += 1
            name = f"{base}-{suffix}"
        return name

    def create_index(self, name: str) -> None:
        if self.client.indices.exists(index=name):
            logger.info("Index %s already exists", name)
            return
        logger.info("Creating index %s", name)
        self.client.indices.create(index=name, body=build_index_body())

    def resolve_write_index(self) -> str:
        """The concrete index the alias points at, creating one if needed.

        A fresh install has neither, so the first call bootstraps both. This is
        what lets the setup instructions be "run the indexer" rather than a
        sequence of curl commands a reader has to get right.
        """
        try:
            aliased = self.client.indices.get_alias(name=self.alias)
        except NotFoundError:
            aliased = {}

        if aliased:
            return next(iter(aliased))

        # An index literally named as the alias would block the alias from ever
        # being created. Refuse rather than half-work.
        if self.client.indices.exists(index=self.alias):
            raise IndexingError(
                f"{self.alias!r} exists as a concrete index, not an alias. "
                f"Rebuilding cannot swap it atomically. Delete it and run the "
                f"indexer with --rebuild."
            )

        name = self.versioned_name()
        self.create_index(name)
        self.point_alias_at(name)
        return name

    def point_alias_at(self, name: str) -> list[str]:
        """Move the alias onto ``name`` atomically, returning what it left.

        Remove and add travel in one ``actions`` payload so there is no instant
        where the alias resolves to nothing. Doing this as two calls would give
        readers a window of "index_not_found_exception" during every rebuild.
        """
        try:
            previous = list(self.client.indices.get_alias(name=self.alias))
        except NotFoundError:
            previous = []

        actions: list[dict[str, Any]] = [
            {"remove": {"index": old, "alias": self.alias}} for old in previous
        ]
        actions.append({"add": {"index": name, "alias": self.alias}})
        self.client.indices.update_aliases(body={"actions": actions})
        logger.info("Alias %s now points at %s (was %s)", self.alias, name, previous or "nothing")
        return previous

    def refresh(self, index: str | None = None) -> None:
        """Make recent writes visible to search.

        OpenSearch refreshes on its own about once a second. Verification and
        tests need to read immediately after writing, and without this they
        race that interval and fail intermittently - which is worse than
        failing consistently.
        """
        self.client.indices.refresh(index=index or self.alias)

    def count(self, index: str | None = None) -> int:
        return int(self.client.count(index=index or self.alias)["count"])

    # -- writing -----------------------------------------------------------

    def drop_extra_chunks(self, index: str, arxiv_ids: list[str]) -> None:
        """Delete documents for chunks beyond the first, for these papers.

        A paper that was sentence-split into three chunks and later revised
        down to one would leave ``#1`` and ``#2`` behind as orphans: nothing
        overwrites them, and they would keep answering queries with text the
        paper no longer contains.

        Only chunk 0 is exempt, and it is exempt deliberately - it always
        exists and is always rewritten, so the paper never stops being
        searchable while this runs.
        """
        if not arxiv_ids:
            return
        self.client.delete_by_query(
            index=index,
            body={
                "query": {
                    "bool": {
                        "filter": [
                            {"terms": {"arxiv_id": arxiv_ids}},
                            {"range": {"chunk_index": {"gte": 1}}},
                        ]
                    }
                }
            },
            # Orphans are rare and a conflict here means something else is
            # writing concurrently; proceeding is safer than aborting the run.
            conflicts="proceed",
        )

    def bulk_index(
        self,
        index: str,
        documents: list[tuple[str, dict[str, Any]]],
    ) -> tuple[int, list[str]]:
        """Index ``(id, source)`` pairs, returning ``(indexed, failures)``.

        **The bulk API returns HTTP 200 even when individual documents fail.**
        Errors live in ``response["items"][n]["index"]["error"]``. Trusting the
        status code is the classic way to end up with an index that is quietly
        missing a slice of the corpus, and the only symptom is a recall number
        that is lower than it should be for no visible reason.
        """
        if not documents:
            return 0, []

        payload: list[dict[str, Any]] = []
        for doc_id, source in documents:
            payload.append({"index": {"_index": index, "_id": doc_id}})
            payload.append(source)

        response = self.client.bulk(body=payload)
        if not response.get("errors"):
            return len(documents), []

        indexed = 0
        failures: list[str] = []
        for item, (doc_id, source) in zip(response["items"], documents, strict=True):
            result = item.get("index", {})
            error = result.get("error")
            if error is None:
                indexed += 1
                continue
            # Retry alone. A batch can fail for reasons that are not this
            # document's fault, and retrying individually is what separates
            # "one malformed record" from "the whole batch was rejected".
            try:
                self.client.index(index=index, id=doc_id, body=source)
                indexed += 1
            except Exception as exc:  # noqa: BLE001 - reported, not raised
                reason = error.get("reason", str(error))
                failures.append(f"{doc_id}: {reason} (retry: {exc})")
                logger.error("Failed to index %s: %s", doc_id, reason)

        return indexed, failures

    # -- reading -----------------------------------------------------------

    def bm25_search(self, query: str, *, size: int = 10) -> list[dict[str, Any]]:
        """Keyword search. Title is boosted because it carries the densest terms."""
        body = {
            "size": size,
            "query": {
                "multi_match": {
                    "query": query,
                    "fields": ["title^2", "abstract", "text"],
                    "type": "best_fields",
                }
            },
        }
        return self.client.search(index=self.alias, body=body)["hits"]["hits"]


def index_corpus(
    *,
    embedder: Embedder | None = None,
    search_index: SearchIndex | None = None,
    index: str | None = None,
    only_stale: bool = True,
    limit: int | None = None,
    batch_size: int = DEFAULT_BULK_SIZE,
    mark_papers: bool = True,
) -> IndexingResult:
    """Embed chunks, store the vectors in PostgreSQL and index the text in OpenSearch.

    ``only_stale`` restricts the pass to chunks that are not fully searchable:
    no ``indexed_at``, or no vector. Ingestion clears ``indexed_at`` on exactly
    the papers whose text changed, so a no-op ingestion leaves nothing to do
    here. The ``embedding`` check catches the other direction - a chunk that
    reached OpenSearch but never got a vector would otherwise stay invisible to
    dense search forever, with nothing flagging it.
    """
    embedder = embedder or get_embedder()
    search_index = search_index or SearchIndex(build_client())
    target = index or search_index.resolve_write_index()

    queryset = Chunk.objects.select_related("paper").order_by("pk")
    if only_stale:
        queryset = queryset.filter(Q(indexed_at__isnull=True) | Q(embedding__isnull=True))
    if limit is not None:
        queryset = queryset[:limit]

    result = IndexingResult(index=target)
    batch: list[Chunk] = []

    def flush(chunks: list[Chunk]) -> None:
        if not chunks:
            return
        vectors = embedder.embed_documents([chunk.text for chunk in chunks])
        if len(vectors) != len(chunks):
            raise IndexingError(
                f"Embedder returned {len(vectors)} vectors for {len(chunks)} chunks"
            )

        # PostgreSQL first. If the process dies here the chunks are still
        # unmarked, so the next run re-embeds and re-indexes them - wasteful but
        # correct. Writing OpenSearch first and dying would leave a document
        # that BM25 can find and dense search cannot, which is the failure that
        # looks like a retrieval quality problem instead of a crash.
        store_vectors(chunks, vectors)

        search_index.drop_extra_chunks(target, [chunk.paper.arxiv_id for chunk in chunks])

        documents = [
            (document_id(chunk.paper.arxiv_id, chunk.chunk_index), build_document(chunk))
            for chunk in chunks
        ]
        indexed, failures = search_index.bulk_index(target, documents)

        result.documents_seen += len(documents)
        result.documents_indexed += indexed
        result.documents_failed += len(failures)
        result.failures.extend(failures)

        if not failures:
            _mark_indexed(chunks, mark_papers=mark_papers, result=result)
        # When part of a batch failed, nothing in it is marked. The chunks stay
        # stale and the next run retries them - re-indexing a document that
        # actually succeeded is free, whereas marking one that failed loses it
        # permanently.

    for chunk in queryset.iterator(chunk_size=batch_size):
        batch.append(chunk)
        if len(batch) >= batch_size:
            flush(batch)
            batch = []
    flush(batch)

    search_index.refresh(target)
    logger.info(
        "Indexed %s/%s documents into %s (%s failures)",
        result.documents_indexed,
        result.documents_seen,
        target,
        result.documents_failed,
    )
    return result


def _mark_indexed(chunks: list[Chunk], *, mark_papers: bool, result: IndexingResult) -> None:
    """Record success in PostgreSQL, after OpenSearch confirmed it.

    The order matters. Marking first and indexing second would mean a crash in
    between leaves rows claiming to be searchable that are not, and nothing
    would ever retry them.
    """
    now = timezone.now()
    for chunk in chunks:
        chunk.indexed_at = now
    Chunk.objects.bulk_update(chunks, ["indexed_at"])

    if not mark_papers:
        return
    paper_ids = {chunk.paper_id for chunk in chunks}
    # Only papers with no remaining unindexed chunk are marked. A paper split
    # across a batch boundary is not searchable in full until its last chunk
    # lands, and claiming otherwise would hide a genuine gap.
    incomplete = set(
        Chunk.objects.filter(paper_id__in=paper_ids, indexed_at__isnull=True).values_list(
            "paper_id", flat=True
        )
    )
    result.papers_marked += Paper.objects.filter(pk__in=paper_ids - incomplete).update(
        indexed_at=now
    )
