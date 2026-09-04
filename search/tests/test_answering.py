"""Tests for the answering layer.

Every test here uses a stub provider. That is not a compromise: the properties
being checked - does the prompt carry numbered passages, does an unrecognised
citation get flagged, does an unavailable provider degrade instead of raising -
are properties of this code, not of the hosted model. Calling a real model
would make each of them slower, non-deterministic and dependent on a secret,
while testing strictly less.
"""

from __future__ import annotations

import pytest
import requests

from papers.models import Paper
from search.answering import (
    ABSTAIN_TOKEN,
    AnswerResult,
    Passage,
    answer,
    build_passages,
    build_user_prompt,
    is_abstention,
    parse_citations,
)
from search.llm import (
    Completion,
    GroqProvider,
    LLMError,
    LLMProvider,
    LLMUnavailable,
    NullProvider,
    get_provider,
)
from search.retrievers.base import RetrievalResult


class StubProvider:
    """Returns whatever it was told to, and remembers what it was asked."""

    name = "stub"
    model = "stub-1"

    def __init__(self, reply: str = "A grounded answer [1].", fail: Exception | None = None):
        self.reply = reply
        self.fail = fail
        self.calls: list[tuple[str, str]] = []

    @property
    def available(self) -> bool:
        return True

    def complete(self, system: str, user: str) -> Completion:
        self.calls.append((system, user))
        if self.fail:
            raise self.fail
        return Completion(
            text=self.reply, model=self.model, prompt_tokens=100, completion_tokens=20
        )


def results(*ids: str) -> list[RetrievalResult]:
    return [RetrievalResult(arxiv_id=i, score=1.0 - n / 10, rank=n + 1) for n, i in enumerate(ids)]


@pytest.fixture
def corpus(db):
    from django.utils import timezone

    made = []
    for n in range(1, 4):
        made.append(
            Paper.objects.create(
                arxiv_id=f"2501.0000{n}",
                version=1,
                title=f"Paper {n}",
                abstract=f"Abstract number {n}. " * 5,
                authors=["A. Author"],
                categories=["hep-ex"],
                primary_category="hep-ex",
                published_at=timezone.now(),
                arxiv_updated_at=timezone.now(),
                abs_url=f"https://arxiv.org/abs/2501.0000{n}",
                pdf_url=f"https://arxiv.org/pdf/2501.0000{n}",
                content_hash=f"hash{n}",
            )
        )
    return made


class TestProtocolConformance:
    def test_stub_satisfies_the_protocol(self):
        assert isinstance(StubProvider(), LLMProvider)

    def test_null_and_groq_satisfy_the_protocol(self):
        assert isinstance(NullProvider(), LLMProvider)
        assert isinstance(GroqProvider(api_key="x"), LLMProvider)


class TestNullProvider:
    def test_reports_itself_unavailable(self):
        assert NullProvider().available is False

    def test_raises_rather_than_returning_empty_text(self):
        # An empty string would be indistinguishable from a model that had
        # nothing to say, and would silently drive the abstention rate to 100%.
        with pytest.raises(LLMUnavailable):
            NullProvider().complete("s", "u")


class TestGroqProvider:
    def test_unavailable_without_a_key(self):
        assert GroqProvider(api_key="").available is False

    def test_get_provider_falls_back_to_null_without_a_key(self, settings):
        settings.GROQ_API_KEY = ""
        assert isinstance(get_provider(), NullProvider)

    def test_get_provider_returns_groq_with_a_key(self, settings):
        settings.GROQ_API_KEY = "gsk_test"
        provider = get_provider()
        assert isinstance(provider, GroqProvider)
        assert provider.available is True

    def test_a_client_error_is_not_retried(self, monkeypatch):
        # A decommissioned model name returns 404 forever. Retrying it burns a
        # minute of backoff to reach the same failure.
        calls = []

        class Response:
            status_code = 404
            text = "model_not_found"
            headers: dict[str, str] = {}

        def post(*args, **kwargs):
            calls.append(1)
            return Response()

        provider = GroqProvider(api_key="k", max_retries=3)
        monkeypatch.setattr(provider.session, "post", post)
        with pytest.raises(LLMError, match="404"):
            provider.complete("s", "u")
        assert len(calls) == 1

    def test_a_rate_limit_is_retried_and_then_succeeds(self, monkeypatch):
        sleeps: list[float] = []
        monkeypatch.setattr("search.llm.time.sleep", lambda s: sleeps.append(s))

        class Limited:
            status_code = 429
            text = "Please try again in 3.5s."
            headers: dict[str, str] = {}

        class Ok:
            status_code = 200
            headers: dict[str, str] = {}

            def json(self):
                return {
                    "choices": [{"message": {"content": "done [1]"}}],
                    "model": "m",
                    "usage": {"prompt_tokens": 5, "completion_tokens": 2},
                }

        responses = [Limited(), Limited(), Ok()]
        provider = GroqProvider(api_key="k", max_retries=3)
        monkeypatch.setattr(provider.session, "post", lambda *a, **k: responses.pop(0))

        completion = provider.complete("s", "u")
        assert completion.text == "done [1]"
        assert completion.total_tokens == 7
        # The wait came from the server's own hint, not a guess.
        assert sleeps == [4.0, 4.0]

    def test_retry_after_header_wins_over_the_prose(self, monkeypatch):
        sleeps: list[float] = []
        monkeypatch.setattr("search.llm.time.sleep", lambda s: sleeps.append(s))

        class Limited:
            status_code = 429
            text = "Please try again in 30s."
            headers = {"retry-after": "2"}

        class Ok:
            status_code = 200
            headers: dict[str, str] = {}

            def json(self):
                return {"choices": [{"message": {"content": "x"}}], "model": "m", "usage": {}}

        responses = [Limited(), Ok()]
        provider = GroqProvider(api_key="k")
        monkeypatch.setattr(provider.session, "post", lambda *a, **k: responses.pop(0))
        provider.complete("s", "u")
        assert sleeps == [2.5]

    def test_retries_are_bounded(self, monkeypatch):
        monkeypatch.setattr("search.llm.time.sleep", lambda s: None)

        class Limited:
            status_code = 429
            text = "try again in 1.0s"
            headers: dict[str, str] = {}

        calls = []

        def post(*args, **kwargs):
            calls.append(1)
            return Limited()

        provider = GroqProvider(api_key="k", max_retries=2)
        monkeypatch.setattr(provider.session, "post", post)
        with pytest.raises(LLMError):
            provider.complete("s", "u")
        assert len(calls) == 3  # the first attempt plus two retries

    def test_a_network_error_becomes_an_llm_error(self, monkeypatch):
        provider = GroqProvider(api_key="k")

        def boom(*args, **kwargs):
            raise requests.ConnectionError("no route")

        monkeypatch.setattr(provider.session, "post", boom)
        with pytest.raises(LLMError, match="no route"):
            provider.complete("s", "u")


class TestPassages:
    def test_numbering_follows_retrieval_rank(self, corpus):
        # Deliberately the reverse of primary-key order, because the database
        # is free to return rows in any order and citation [1] must still mean
        # the top result.
        passages = build_passages(results("2501.00003", "2501.00001", "2501.00002"))
        assert [p.number for p in passages] == [1, 2, 3]
        assert [p.arxiv_id for p in passages] == ["2501.00003", "2501.00001", "2501.00002"]

    def test_context_size_truncates(self, corpus):
        assert len(build_passages(results("2501.00001", "2501.00002", "2501.00003"), 2)) == 2

    def test_a_paper_missing_from_the_database_is_skipped_not_faked(self, corpus):
        passages = build_passages(results("2501.00001", "9999.99999", "2501.00002"))
        assert [p.arxiv_id for p in passages] == ["2501.00001", "2501.00002"]
        # Renumbered, so there is no gap the model could cite into.
        assert [p.number for p in passages] == [1, 2]

    def test_no_results_gives_no_passages(self, corpus):
        assert build_passages([]) == []


class TestPrompt:
    def test_every_passage_is_numbered_in_the_prompt(self):
        passages = [
            Passage(1, "a", "First", "text one"),
            Passage(2, "b", "Second", "text two"),
        ]
        prompt = build_user_prompt("why", passages)
        assert "[1] First" in prompt
        assert "[2] Second" in prompt

    def test_the_question_comes_after_the_passages(self):
        prompt = build_user_prompt("why", [Passage(1, "a", "T", "body")])
        assert prompt.index("body") < prompt.index("Question: why")

    def test_the_abstention_instruction_is_restated_at_the_end(self):
        prompt = build_user_prompt("why", [Passage(1, "a", "T", "body")])
        assert prompt.rstrip().endswith(f"reply with exactly {ABSTAIN_TOKEN}.")


class TestCitationParsing:
    def test_valid_citations_are_kept_in_order(self):
        cited, invalid = parse_citations("Claim [2] and claim [1].", {1, 2, 3})
        assert cited == [2, 1]
        assert invalid == []

    def test_a_citation_to_a_passage_that_was_never_supplied_is_flagged(self):
        cited, invalid = parse_citations("As shown [7].", {1, 2, 3})
        assert invalid == [7]

    def test_repeated_fabrications_are_counted_repeatedly(self):
        # Collapsing them to one would flatter the precision figure: three
        # references to a passage that does not exist is three errors.
        _, invalid = parse_citations("[7] and [7] and [7]", {1})
        assert invalid == [7, 7, 7]

    def test_no_citations_is_not_an_error(self):
        assert parse_citations("A bare assertion.", {1}) == ([], [])


class TestAbstentionDetection:
    def test_the_bare_token_counts(self):
        assert is_abstention(ABSTAIN_TOKEN)

    def test_surrounding_whitespace_and_formatting_are_tolerated(self):
        assert is_abstention(f"  `{ABSTAIN_TOKEN}`  ")
        assert is_abstention(f"{ABSTAIN_TOKEN}\n")

    def test_an_answer_that_merely_mentions_the_token_is_not_an_abstention(self):
        # A model that answers at length and then hedges has not refused.
        assert not is_abstention(f"The cross section is 3 pb [1]. Otherwise {ABSTAIN_TOKEN}.")

    def test_empty_text_is_not_an_abstention(self):
        assert not is_abstention("   ")


class TestAnswer:
    def test_a_grounded_answer_is_parsed(self, corpus):
        provider = StubProvider("The result follows [1] and [2].")
        result = answer("q", results("2501.00001", "2501.00002"), provider=provider)
        assert result.abstained is False
        assert result.citations == [1, 2]
        assert result.invalid_citations == []
        assert result.citation_precision == 1.0
        assert result.is_grounded is True
        assert result.cited_ids == ["2501.00001", "2501.00002"]

    def test_a_fabricated_citation_lowers_precision(self, corpus):
        provider = StubProvider("Claim [1] and claim [9].")
        result = answer("q", results("2501.00001", "2501.00002"), provider=provider)
        assert result.invalid_citations == [9]
        assert result.citation_precision == 0.5
        assert result.is_grounded is False
        # The fabricated number resolves to nothing, so it cannot leak into the
        # list of papers shown to a user as sources.
        assert result.cited_ids == ["2501.00001"]

    def test_an_abstention_is_recorded_without_citations(self, corpus):
        provider = StubProvider(ABSTAIN_TOKEN)
        result = answer("q", results("2501.00001"), provider=provider)
        assert result.abstained is True
        assert result.citations == []
        # Not 0.0: an abstention cites nothing, and scoring that as perfectly
        # wrong would punish the behaviour the system is trying to encourage.
        assert result.citation_precision is None

    def test_an_uncited_answer_is_not_grounded(self, corpus):
        result = answer("q", results("2501.00001"), provider=StubProvider("Trust me."))
        assert result.is_grounded is False
        assert result.citation_precision is None

    def test_no_results_abstains_without_calling_the_model(self, corpus):
        provider = StubProvider()
        result = answer("q", [], provider=provider)
        assert result.abstained is True
        assert result.degraded is False
        assert provider.calls == []
        assert "no results" in result.reason

    def test_an_unavailable_provider_degrades_rather_than_raising(self, corpus):
        result = answer("q", results("2501.00001"), provider=NullProvider())
        assert result.degraded is True
        # Crucially not an abstention: the system did not decide to refuse, it
        # was unable to try, and conflating the two would inflate the headline
        # abstention rate with configuration problems.
        assert result.abstained is False
        assert "no LLM provider" in result.reason

    def test_a_provider_failure_degrades_and_keeps_the_reason(self, corpus):
        provider = StubProvider(fail=LLMError("Groq returned 500"))
        result = answer("q", results("2501.00001"), provider=provider)
        assert result.degraded is True
        assert result.abstained is False
        assert "500" in result.reason

    def test_context_size_limits_what_the_model_sees(self, corpus):
        provider = StubProvider()
        answer("q", results("2501.00001", "2501.00002", "2501.00003"), provider, context_size=2)
        _, user = provider.calls[0]
        assert "[2]" in user
        assert "[3]" not in user

    def test_usage_is_carried_through(self, corpus):
        result = answer("q", results("2501.00001"), provider=StubProvider())
        assert result.usage == {"prompt_tokens": 100, "completion_tokens": 20}


class TestCitedIds:
    def test_duplicate_citations_yield_one_source(self):
        result = AnswerResult(
            query="q",
            text="[1] and again [1]",
            abstained=False,
            passages=[Passage(1, "2501.1", "T", "b")],
            citations=[1, 1],
            invalid_citations=[],
            degraded=False,
            model="m",
            elapsed_ms=1,
        )
        # Precision still counts both, because both were emitted; the source
        # list is deduplicated because a reader wants papers, not mentions.
        assert result.cited_ids == ["2501.1"]
        assert result.citation_precision == 1.0
