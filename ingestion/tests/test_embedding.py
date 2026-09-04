"""Tests for the embedding layer.

None of these load a real model. What is worth testing here is our code: that
vectors come back normalised, that the query prefix goes on queries and only on
queries, that a dimension mismatch is caught with a message that names the real
problem, and that the model is loaded once rather than per call.

Testing the model itself would be testing somebody else's code, would need a
network and several hundred megabytes, and would tell us nothing about whether
this project uses it correctly.
"""

from __future__ import annotations

import math
import sys

import pytest

from ingestion import embedding
from ingestion.embedding import (
    QUERY_PREFIX,
    EmbeddingError,
    HashingEmbedder,
    SentenceTransformerEmbedder,
)


def norm(vector: list[float]) -> float:
    return math.sqrt(sum(value * value for value in vector))


def dot(left: list[float], right: list[float]) -> float:
    return sum(a * b for a, b in zip(left, right, strict=True))


class TestHashingEmbedder:
    """The stand-in has to be a fair stand-in, or tests built on it lie."""

    def test_vectors_are_unit_length(self) -> None:
        vectors = HashingEmbedder(dimension=16).embed_documents(
            ["a search for long-lived particles", "measurement of the top quark mass"]
        )
        for vector in vectors:
            assert norm(vector) == pytest.approx(1.0)

    def test_is_deterministic(self) -> None:
        first = HashingEmbedder(dimension=16).embed_documents(["neutrino oscillation"])
        second = HashingEmbedder(dimension=16).embed_documents(["neutrino oscillation"])
        assert first == second

    def test_shared_vocabulary_scores_higher(self) -> None:
        """Similar text must actually produce similar vectors.

        Without this the fake would be indistinguishable from random noise, and
        every test that asserts anything about retrieval order would pass for
        the wrong reason.
        """
        model = HashingEmbedder(dimension=256)
        related, unrelated = model.embed_documents(
            [
                "search for long-lived particles in proton collisions",
                "a study of medieval agricultural practice",
            ]
        )
        (target,) = model.embed_documents(["search for long-lived particles at the collider"])
        assert dot(target, related) > dot(target, unrelated)

    def test_empty_text_gets_a_valid_unit_vector(self) -> None:
        """A zero vector has no direction, so cosine against it is undefined.

        Letting one into the index poisons every kNN query that touches it with
        a NaN score rather than failing anywhere near the cause.
        """
        (vector,) = HashingEmbedder(dimension=16).embed_documents(["   "])
        assert norm(vector) == pytest.approx(1.0)

    def test_respects_requested_dimension(self) -> None:
        (vector,) = HashingEmbedder(dimension=384).embed_documents(["anything"])
        assert len(vector) == 384

    def test_no_documents_means_no_call(self) -> None:
        assert HashingEmbedder().embed_documents([]) == []


class TestQueryPrefix:
    """The asymmetry BGE was trained with, pinned so it cannot drift."""

    def test_query_is_prefixed(self) -> None:
        model = HashingEmbedder(dimension=64)
        query = model.embed_query("higgs boson decay")
        (prefixed,) = model.embed_documents([QUERY_PREFIX + "higgs boson decay"])
        assert query == prefixed

    def test_document_is_not_prefixed(self) -> None:
        model = HashingEmbedder(dimension=64)
        (document,) = model.embed_documents(["higgs boson decay"])
        assert model.embed_query("higgs boson decay") != document

    def test_prefix_text_is_the_published_one(self) -> None:
        # Pinned as a literal. A paraphrase would still run, still index and
        # still return results - just slightly worse ones, with nothing in any
        # log to say so.
        assert QUERY_PREFIX == "Represent this sentence for searching relevant passages: "


class TestModelLoading:
    def test_missing_package_explains_where_it_lives(self, monkeypatch) -> None:
        """The failure has to name the fix.

        sentence-transformers is deliberately not in base.txt, so "No module
        named sentence_transformers" is a confusing message for anyone who has
        already installed the requirements.
        """
        monkeypatch.setitem(sys.modules, "sentence_transformers", None)
        monkeypatch.setattr(embedding, "_MODEL_CACHE", {})
        with pytest.raises(EmbeddingError, match="requirements/ml.txt"):
            embedding._load_model("BAAI/bge-small-en-v1.5", "cpu")

    def test_model_is_loaded_once(self, monkeypatch) -> None:
        loads: list[str] = []

        class StubModel:
            def get_sentence_embedding_dimension(self) -> int:
                return 384

        def fake_load(model_name: str, device: str) -> StubModel:
            loads.append(model_name)
            return StubModel()

        monkeypatch.setattr(embedding, "_load_model", fake_load)
        embedder = SentenceTransformerEmbedder(expected_dim=384)
        first = embedder.model
        second = embedder.model
        assert first is second
        assert loads == ["BAAI/bge-small-en-v1.5"]

    def test_dimension_mismatch_is_caught_at_load(self, monkeypatch) -> None:
        """Caught here, where the message can name the model and the setting.

        Left uncaught, the same mistake surfaces as an indexing rejection that
        talks about the ``embedding`` field, several layers away from the cause.
        """

        class WrongSizeModel:
            def get_sentence_embedding_dimension(self) -> int:
                return 768

        monkeypatch.setattr(embedding, "_load_model", lambda name, device: WrongSizeModel())
        embedder = SentenceTransformerEmbedder(expected_dim=384)
        with pytest.raises(EmbeddingError, match="768.*384"):
            _ = embedder.model

    def test_device_defaults_to_cpu(self) -> None:
        # Not cosmetic. The deployment target has no GPU, so a machine-dependent
        # default would mean the laptop and the VM run different code paths.
        assert SentenceTransformerEmbedder().device == "cpu"


class TestTokenCounterFallback:
    """get_token_counter must degrade rather than explode.

    Chunking runs in CI, where requirements/ml.txt is not installed. If the
    absence of PyTorch raised here, ingestion would be impossible to test
    without a 2 GB dependency.
    """

    def test_returns_none_when_sentence_transformers_is_missing(self, monkeypatch) -> None:
        monkeypatch.setattr(embedding, "_MODEL_CACHE", {})
        monkeypatch.setitem(sys.modules, "sentence_transformers", None)
        assert embedding.get_token_counter() is None
