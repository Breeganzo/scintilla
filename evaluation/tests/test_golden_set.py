"""Tests for the golden set file and its loader.

Two kinds of test live here. The first kind checks the loader rejects malformed
input - ordinary defensive testing. The second kind checks the *shipped* golden
set itself, which is unusual: it treats a data file as something that can
regress. It can. A hand-edited YAML file with fifty entries is exactly the kind
of artefact that acquires a duplicate id or an unquoted number during a hurried
edit, and every one of those failures is silent at runtime.
"""

import textwrap

import pytest
import yaml

from evaluation.golden import (
    QUERIES_PER_CLASS,
    QUERY_CLASSES,
    GoldenSetError,
    load_golden_set,
)

MINIMAL = {
    "version": 1,
    "queries": [
        {"id": "a_1", "class": "exact_term", "query": "q", "relevant_ids": ["2601.00001"]},
    ],
}


def write(tmp_path, data):
    path = tmp_path / "test.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


def full_set():
    """A structurally valid set: ten queries in each of the five classes."""
    queries = []
    for query_class in QUERY_CLASSES:
        for i in range(QUERIES_PER_CLASS):
            queries.append(
                {
                    "id": f"{query_class}_{i}",
                    "class": query_class,
                    "query": f"a query about {i}",
                    "relevant_ids": [] if query_class == "unanswerable" else [f"260{i}.0000{i}"],
                }
            )
    return {"version": 1, "queries": queries}


class TestLoading:
    def test_a_valid_set_loads(self, tmp_path):
        golden = load_golden_set(path=write(tmp_path, full_set()))
        assert len(golden) == len(QUERY_CLASSES) * QUERIES_PER_CLASS

    def test_a_missing_file_is_an_error(self, tmp_path):
        with pytest.raises(GoldenSetError, match="no golden set"):
            load_golden_set(path=tmp_path / "absent.yaml")

    def test_by_class_filters(self, tmp_path):
        golden = load_golden_set(path=write(tmp_path, full_set()))
        assert len(golden.by_class("paraphrase")) == QUERIES_PER_CLASS

    def test_relevant_ids_collects_across_queries(self, tmp_path):
        golden = load_golden_set(path=write(tmp_path, full_set()))
        # Four answerable classes share the same ten synthetic ids.
        assert len(golden.relevant_ids) == QUERIES_PER_CLASS


class TestValidation:
    def test_a_duplicate_id_is_rejected(self, tmp_path):
        data = full_set()
        data["queries"][1]["id"] = data["queries"][0]["id"]
        with pytest.raises(GoldenSetError, match="duplicate query id"):
            load_golden_set(path=write(tmp_path, data))

    def test_an_unknown_class_is_rejected(self, tmp_path):
        data = full_set()
        data["queries"][0]["class"] = "vibes"
        with pytest.raises(GoldenSetError, match="unknown class"):
            load_golden_set(path=write(tmp_path, data))

    def test_an_empty_query_is_rejected(self, tmp_path):
        data = full_set()
        data["queries"][0]["query"] = "   "
        with pytest.raises(GoldenSetError, match="empty query"):
            load_golden_set(path=write(tmp_path, data))

    def test_a_repeated_relevant_id_is_rejected(self, tmp_path):
        data = full_set()
        data["queries"][0]["relevant_ids"] = ["2601.00001", "2601.00001"]
        with pytest.raises(GoldenSetError, match="same paper twice"):
            load_golden_set(path=write(tmp_path, data))

    def test_an_unanswerable_query_may_not_have_labels(self, tmp_path):
        """The class only means anything if it is empty. A stray label here
        would convert a test of abstention into a test of retrieval."""
        data = full_set()
        for query in data["queries"]:
            if query["class"] == "unanswerable":
                query["relevant_ids"] = ["2601.00001"]
                break
        with pytest.raises(GoldenSetError, match="unanswerable .* has relevant ids"):
            load_golden_set(path=write(tmp_path, data))

    def test_an_answerable_query_must_have_labels(self, tmp_path):
        data = full_set()
        data["queries"][0]["relevant_ids"] = []
        with pytest.raises(GoldenSetError, match="no relevant ids"):
            load_golden_set(path=write(tmp_path, data))

    def test_an_unbalanced_class_is_rejected(self, tmp_path):
        data = full_set()
        data["queries"] = data["queries"][1:]
        with pytest.raises(GoldenSetError, match="expected 10"):
            load_golden_set(path=write(tmp_path, data))

    def test_a_missing_field_is_rejected(self, tmp_path):
        data = full_set()
        del data["queries"][0]["relevant_ids"]
        with pytest.raises(GoldenSetError, match="missing"):
            load_golden_set(path=write(tmp_path, data))

    def test_a_malformed_arxiv_id_is_rejected(self, tmp_path):
        data = full_set()
        data["queries"][0]["relevant_ids"] = ["not-an-id"]
        with pytest.raises(GoldenSetError, match="malformed arXiv id"):
            load_golden_set(path=write(tmp_path, data))


class TestUnquotedIdentifiers:
    """The specific corruption this project is most likely to suffer.

    In YAML, 2607.07800 without quotes is the float 2607.078. It parses without
    complaint, survives review because it looks like an arXiv id, and then
    matches nothing - so the query scores zero for every retriever equally,
    which in an ablation table is indistinguishable from a hard query.
    """

    def test_yaml_really_does_do_this(self, tmp_path):
        path = tmp_path / "proof.yaml"
        path.write_text("ids: [2607.07800]\n", encoding="utf-8")
        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))["ids"][0]
        assert loaded == pytest.approx(2607.078)
        assert not isinstance(loaded, str)

    def test_the_loader_catches_it(self, tmp_path):
        path = tmp_path / "unquoted.yaml"
        path.write_text(
            textwrap.dedent(
                """\
                version: 1
                queries:
                  - id: a_1
                    class: exact_term
                    query: q
                    relevant_ids: [2607.07800]
                """
            ),
            encoding="utf-8",
        )
        with pytest.raises(GoldenSetError, match="must be quoted"):
            load_golden_set(path=path)


class TestShippedGoldenSet:
    """Assertions about the real v1.yaml, not a fixture."""

    @pytest.fixture(scope="class")
    def golden(self):
        return load_golden_set()

    def test_it_loads_and_validates(self, golden):
        assert len(golden) == 50

    def test_every_class_is_present_and_balanced(self, golden):
        for query_class in QUERY_CLASSES:
            assert len(golden.by_class(query_class)) == QUERIES_PER_CLASS

    def test_the_corpus_snapshot_is_recorded(self, golden):
        """Labels are only valid against the collection they were made on."""
        assert golden.corpus["papers"] > 0
        assert golden.corpus["chunks"] > 0
        assert golden.corpus["captured"]

    def test_the_pooling_method_is_recorded(self, golden):
        assert set(golden.pool["retrievers"]) == {"bm25", "dense"}
        assert golden.pool["depth"] >= 10

    def test_every_answerable_query_is_documented(self, golden):
        """Notes are where a borderline call is defensible rather than arbitrary."""
        undocumented = [q.id for q in golden if not q.notes]
        assert undocumented == []


@pytest.mark.django_db
class TestLabelsMatchTheCorpus:
    """Checks that need the collection, not just the file.

    These skip on an empty database rather than fail. The unit suite runs
    against a fresh test database in which every label is trivially missing, so
    failing here would mean a permanently red suite that teaches nobody
    anything. The real check is `manage.py validate_golden_set`, which runs
    against the live corpus and is a required step before publishing a number.
    """

    @pytest.fixture
    def corpus(self):
        from papers.models import Paper

        if not Paper.objects.exists():
            pytest.skip("no corpus loaded; use manage.py validate_golden_set instead")
        return Paper

    def test_relevant_ids_are_wellformed_and_unique_per_query(self):
        """Cheap structural check that runs without a populated database."""
        golden = load_golden_set()
        for query in golden:
            assert len(set(query.relevant_ids)) == len(query.relevant_ids)

    def test_every_labelled_paper_exists(self, corpus):
        """A label pointing at a paper that is not in the collection scores
        zero forever and is indistinguishable from a genuinely hard query."""
        golden = load_golden_set()
        known = set(
            corpus.objects.filter(arxiv_id__in=golden.relevant_ids).values_list(
                "arxiv_id", flat=True
            )
        )
        missing = sorted(golden.relevant_ids - known)
        assert missing == [], f"{len(missing)} labelled papers are not in the corpus"

    def test_no_query_quotes_a_relevant_title_verbatim(self, corpus):
        """A query containing its answer's title is a gift to lexical retrieval.

        It would inflate BM25 for reasons that have nothing to do with whether
        BM25 is any good, and the exact_term class would stop measuring
        vocabulary matching and start measuring copy-paste.
        """
        golden = load_golden_set()
        titles = dict(
            corpus.objects.filter(arxiv_id__in=golden.relevant_ids).values_list("arxiv_id", "title")
        )
        offenders = [
            (query.id, paper_id)
            for query in golden
            for paper_id in query.relevant_ids
            if (title := titles.get(paper_id))
            and len(title) > 20
            and title.lower() in query.query.lower()
        ]
        assert offenders == []
