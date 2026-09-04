"""Turning text into vectors.

**The model: ``BAAI/bge-small-en-v1.5``, 384 dimensions, on CPU.**

Chosen because it is small enough to run on a free-tier ARM VM with no GPU and
still scores well on BEIR, and because 384 dimensions keeps the HNSW graph
small. A larger model would score marginally better and would not fit the
deployment target, which makes it the wrong model regardless of its score.

**Three details that are easy to get wrong and invisible when you do.**

1. **L2-normalise the output.** With unit vectors, cosine similarity reduces to
   a dot product. Skipping this does not error - it silently changes what
   "nearest" means, because vector magnitude then acts as an unintended
   relevance prior favouring longer text.

2. **BGE expects an instruction prefix on queries and not on documents.** The
   model was trained with that asymmetry, so applying the prefix to both sides,
   or to neither, degrades retrieval rather than breaking it. There is no
   exception, no log line and no failing test - the results simply get slightly
   worse. This is the single clearest argument for building the evaluation
   harness in Phase 3: without measurement, a mistake of exactly this shape is
   undetectable.

3. **Load the model once.** Loading takes seconds; encoding takes milliseconds.
   Loading per call would make the cost of the wrong design look like the cost
   of embedding.

**Why this module imports ``sentence_transformers`` lazily.**

The package pulls in PyTorch, which is hundreds of megabytes. Importing it at
module scope would mean the test suite, the linter and every ``manage.py``
invocation paid for it, and CI would install it to run tests that never embed
anything. The same reasoning that keeps the network out of the test suite keeps
the model out of it: tests use :class:`HashingEmbedder`, which is deterministic,
dependency-free and produces genuinely similar vectors for genuinely similar
text, so tests of the *pipeline* stay honest without testing the *model*.
"""

from __future__ import annotations

import hashlib
import logging
import math
from collections.abc import Callable
from typing import Protocol, runtime_checkable

from django.conf import settings

logger = logging.getLogger(__name__)

# Prepended to queries only. See point 2 in the module docstring - this is the
# published prefix for the bge-*-en-v1.5 family and changing it silently costs
# retrieval quality.
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

# Large enough to keep the CPU busy, small enough that a batch of long
# abstracts does not spike memory on a 1 GB VM.
DEFAULT_BATCH_SIZE = 32

# Keyed by (model name, device) so the model survives across calls within a
# process. Airflow tasks and the web worker each pay the load cost once.
_MODEL_CACHE: dict[tuple[str, str], object] = {}


class EmbeddingError(RuntimeError):
    """Raised when a model cannot be loaded or produces the wrong shape."""


@runtime_checkable
class Embedder(Protocol):
    """What the rest of the codebase is allowed to assume about an embedder.

    Defined as a protocol so the indexer depends on this shape rather than on
    sentence-transformers. That is what makes the fake below a drop-in
    replacement and what would make swapping the model a contained change.
    """

    model_name: str

    @property
    def dimension(self) -> int: ...

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


def _load_model(model_name: str, device: str) -> object:
    key = (model_name, device)
    cached = _MODEL_CACHE.get(key)
    if cached is not None:
        return cached

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise EmbeddingError(
            "sentence-transformers is not installed. It lives in "
            "requirements/ml.txt rather than base.txt because it pulls in "
            "PyTorch; install it with: pip install -r requirements/ml.txt"
        ) from exc

    logger.info("Loading embedding model %s onto %s", model_name, device)
    model = SentenceTransformer(model_name, device=device)
    _MODEL_CACHE[key] = model
    return model


def _embedding_dimension(model: object) -> int | None:
    """Ask a SentenceTransformer how wide its vectors are.

    sentence-transformers 5.x renamed ``get_sentence_embedding_dimension`` to
    ``get_embedding_dimension`` and the old name now emits a FutureWarning.
    requirements/ml.txt allows >=3.3,<6.0, so both names have to work; asking
    for the new one first means the pin can widen later without a code change.
    """
    for name in ("get_embedding_dimension", "get_sentence_embedding_dimension"):
        method = getattr(model, name, None)
        if callable(method):
            return method()
    return None


class SentenceTransformerEmbedder:
    """The real embedder.

    ``device`` defaults to CPU explicitly rather than letting the library pick.
    The deployment target is an ARM VM with no GPU, so a machine-dependent
    default would mean the thing that works on a developer laptop behaves
    differently in production - and the difference would surface as a crash on
    deploy rather than as a failing test.
    """

    def __init__(
        self,
        model_name: str | None = None,
        *,
        device: str = "cpu",
        batch_size: int = DEFAULT_BATCH_SIZE,
        expected_dim: int | None = None,
    ) -> None:
        self.model_name = model_name or settings.EMBEDDING_MODEL
        self.device = device
        self.batch_size = batch_size
        self._expected_dim = (
            expected_dim if expected_dim is not None else int(settings.EMBEDDING_DIM)
        )
        self._model: object | None = None

    @property
    def dimension(self) -> int:
        return self._expected_dim

    @property
    def model(self) -> object:
        """The loaded model, with its dimension checked once on first use.

        The check exists because the failure it catches is otherwise
        excruciating: the OpenSearch mapping fixes the vector width at index
        creation, so a model producing 768 where the mapping says 384 fails at
        *indexing* time with an error that talks about the field, not the
        model. Asserting here names the actual problem.
        """
        if self._model is None:
            model = _load_model(self.model_name, self.device)
            actual = _embedding_dimension(model)
            if actual != self._expected_dim:
                raise EmbeddingError(
                    f"{self.model_name} produces {actual}-dimensional vectors but "
                    f"EMBEDDING_DIM is {self._expected_dim}. The OpenSearch mapping "
                    f"is built from EMBEDDING_DIM and cannot be changed in place, "
                    f"so fix the setting and rebuild the index."
                )
            self._model = model
        return self._model

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Encode documents. No prefix - see point 2 in the module docstring."""
        if not texts:
            return []
        vectors = self.model.encode(
            list(texts),
            batch_size=self.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return [[float(value) for value in row] for row in vectors]

    def embed_query(self, text: str) -> list[float]:
        """Encode one query, *with* the instruction prefix."""
        vectors = self.model.encode(
            [QUERY_PREFIX + text],
            batch_size=1,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return [float(value) for value in vectors[0]]

    def count_tokens(self, text: str) -> int:
        """Exact subword token count, replacing the words x 1.3 estimate.

        ``chunking.estimate_tokens`` deliberately over-estimates so that it
        never lets the model truncate silently. Once the model is loaded its
        own tokenizer is available and there is no reason to guess, so the
        recorded ``token_count`` becomes the real number.
        """
        tokenizer = self.model.tokenizer
        return len(tokenizer.encode(text, add_special_tokens=True))


class HashingEmbedder:
    """A deterministic, dependency-free embedder used by the tests.

    This is a signed hashing trick, not a toy: every word is hashed to a bucket
    and a sign, and the resulting bag-of-words vector is L2-normalised. Two
    texts sharing vocabulary genuinely score higher against each other than two
    that do not, so tests can assert on relative similarity and mean it.

    It exists so the test suite never downloads a model, never needs PyTorch
    and runs in milliseconds. Tests written against the real model would be
    testing the model, which is not this project's code, and would fail on a
    machine with no network - the same trap the arXiv fixtures avoid.
    """

    model_name = "hashing-test-embedder"

    def __init__(self, dimension: int = 384) -> None:
        self._dimension = dimension

    @property
    def dimension(self) -> int:
        return self._dimension

    def _vector(self, text: str) -> list[float]:
        buckets = [0.0] * self._dimension
        for word in text.lower().split():
            digest = hashlib.sha256(word.encode("utf-8")).digest()
            bucket = int.from_bytes(digest[:4], "big") % self._dimension
            buckets[bucket] += 1.0 if digest[4] % 2 == 0 else -1.0

        norm = math.sqrt(sum(value * value for value in buckets))
        if norm == 0.0:
            # Empty or whitespace-only input. A zero vector has no direction,
            # which makes cosine similarity undefined, so return a valid unit
            # vector instead of propagating a NaN into the index.
            buckets[0] = 1.0
            return buckets
        return [value / norm for value in buckets]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vector(QUERY_PREFIX + text)

    def count_tokens(self, text: str) -> int:
        return len(text.split())


def get_embedder() -> Embedder:
    """The embedder configured for this environment."""
    return SentenceTransformerEmbedder()


def get_token_counter() -> Callable[[str], int] | None:
    """The real tokenizer, or ``None`` when sentence-transformers is absent.

    Chunking needs to count tokens the way the embedding model counts them,
    but chunking must also work in CI, where requirements/ml.txt is not
    installed. Returning ``None`` lets the caller fall back to the heuristic
    rather than forcing every consumer to catch :class:`EmbeddingError`.
    """
    try:
        embedder = SentenceTransformerEmbedder()
        # Force the load here. The model is lazy, so without this the ImportError
        # would surface on the first call deep inside chunking, long past the
        # point where falling back is still an option.
        _ = embedder.model
        return embedder.count_tokens
    except EmbeddingError:
        logger.warning(
            "sentence-transformers unavailable; chunking will estimate token "
            "counts from whitespace, which under-counts LaTeX-heavy abstracts"
        )
        return None
