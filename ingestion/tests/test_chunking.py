"""Tests for the chunking strategy."""

from __future__ import annotations

from ingestion.chunking import (
    build_document_text,
    chunk_paper,
    estimate_tokens,
    split_sentences,
)

TITLE = "Measurement of the Higgs boson coupling to tau leptons"
ABSTRACT = (
    "We present a measurement of the Higgs boson coupling to tau leptons using "
    "proton-proton collision data collected at a centre-of-mass energy of 13 TeV. "
    "The observed significance is 5.2 standard deviations."
)


class TestDocumentText:
    def test_title_and_abstract_are_joined_with_a_blank_line(self):
        text = build_document_text(TITLE, ABSTRACT)
        assert text == f"{TITLE}\n\n{ABSTRACT}"

    def test_the_title_is_included_not_discarded(self):
        # The title carries detector, particle and collaboration names that BM25
        # weights heavily and that the abstract sometimes never repeats.
        assert "Higgs boson coupling to tau leptons" in build_document_text(TITLE, ABSTRACT)

    def test_whitespace_is_normalised(self):
        text = build_document_text("A   title\nwrapped", "An\n  abstract")
        assert text == "A title wrapped\n\nAn abstract"


class TestEstimateTokens:
    def test_over_estimates_rather_than_under(self):
        # Erring high chunks slightly early, which is harmless. Erring low lets
        # the model truncate, which is the failure being avoided.
        assert estimate_tokens("one two three four") > 4

    def test_empty_text_is_zero(self):
        assert estimate_tokens("") == 0


class TestSingleChunk:
    def test_a_normal_abstract_produces_exactly_one_chunk(self):
        chunks = chunk_paper(TITLE, ABSTRACT)
        assert len(chunks) == 1
        assert chunks[0].index == 0

    def test_the_chunk_holds_the_whole_document(self):
        chunks = chunk_paper(TITLE, ABSTRACT)
        assert chunks[0].text == build_document_text(TITLE, ABSTRACT)

    def test_token_count_is_recorded(self):
        chunks = chunk_paper(TITLE, ABSTRACT)
        assert chunks[0].token_count == estimate_tokens(chunks[0].text)

    def test_chunking_is_deterministic(self):
        # Non-deterministic chunking would make the content hash unstable and
        # every run would re-embed everything.
        assert chunk_paper(TITLE, ABSTRACT) == chunk_paper(TITLE, ABSTRACT)


class TestOverlongAbstracts:
    def _long_abstract(self, sentences: int = 200) -> str:
        return " ".join(
            f"Sentence number {i} reports a measurement of the cross section."
            for i in range(sentences)
        )

    def test_an_over_long_abstract_is_split(self):
        chunks = chunk_paper(TITLE, self._long_abstract())
        assert len(chunks) > 1

    def test_no_chunk_exceeds_the_ceiling(self):
        chunks = chunk_paper(TITLE, self._long_abstract(), max_tokens=200)
        assert all(chunk.token_count <= 200 for chunk in chunks)

    def test_chunk_indexes_are_contiguous_from_zero(self):
        chunks = chunk_paper(TITLE, self._long_abstract(), max_tokens=200)
        assert [chunk.index for chunk in chunks] == list(range(len(chunks)))

    def test_consecutive_chunks_overlap_by_a_sentence(self):
        chunks = chunk_paper(TITLE, self._long_abstract(), max_tokens=200, overlap_sentences=1)
        first_sentences = split_sentences(chunks[0].text)
        second_sentences = split_sentences(chunks[1].text)
        assert first_sentences[-1] == second_sentences[0]

    def test_no_content_is_lost(self):
        long_abstract = self._long_abstract(60)
        chunks = chunk_paper(TITLE, long_abstract, max_tokens=200)
        joined = " ".join(chunk.text for chunk in chunks)
        for sentence in split_sentences(build_document_text(TITLE, long_abstract)):
            assert sentence in joined

    def test_a_single_unsplittable_sentence_is_emitted_whole(self):
        # Nothing sensible can be done here, but it must not crash or silently
        # drop the paper. It logs loudly instead.
        monster = "word " * 2000
        chunks = chunk_paper(TITLE, monster, max_tokens=100)
        assert len(chunks) == 1
        assert chunks[0].text.startswith(TITLE)


class TestInjectedTokenCounter:
    """The embedding model's tokenizer must be able to drive chunk boundaries.

    The whitespace heuristic under-counted 94% of a 2,975-paper sample and let
    69 abstracts past the 512-token ceiling, where the model truncated them
    silently. These tests pin the fix: a caller can supply the real counter,
    and it is what the ceiling is actually enforced against.
    """

    def test_injected_counter_is_used_for_token_count(self) -> None:
        chunks = chunk_paper(TITLE, ABSTRACT, count_tokens=lambda text: len(text))
        assert chunks[0].token_count == len(chunks[0].text)

    def test_injected_counter_triggers_the_split_the_heuristic_would_miss(self) -> None:
        # The heuristic sees a short document and returns one chunk.
        assert len(chunk_paper(TITLE, ABSTRACT)) == 1

        # A tokenizer that counts characters sees the same text as far over the
        # ceiling, and the split happens. This is the 69-paper case in
        # miniature: same input, different tokenizer, different answer.
        chunks = chunk_paper(TITLE, ABSTRACT, max_tokens=120, count_tokens=lambda text: len(text))
        assert len(chunks) > 1

    def test_defaults_to_the_heuristic_when_no_counter_is_given(self) -> None:
        assert chunk_paper(TITLE, ABSTRACT)[0].token_count == estimate_tokens(
            build_document_text(TITLE, ABSTRACT)
        )
