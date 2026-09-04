"""The explanation must fail closed. These tests are mostly about that.

Every malformed shape below is something a serializer change could plausibly
produce. The requirement is not that the function copes gracefully - it is that
it refuses to describe a ranking it cannot actually read.
"""

import pytest

from mcp_server.explain import explain

_UNKNOWN = "No ranking explanation available for this result."


def test_fused_result_reports_each_retriever():
    text = explain(
        {
            "rrf_k": 60,
            "runs": {
                "bm25": {"rank": 1, "score": 9.6292, "contribution": 0.016393},
                "dense": {"rank": 3, "score": 0.71, "contribution": 0.015873},
            },
        }
    )
    assert "bm25 ranked it #1" in text
    assert "dense ranked it #3" in text
    assert "0.01639" in text
    assert "k=60" in text


def test_single_retriever_hit_is_called_out():
    # The most important sentence this function can produce. One retriever
    # finding a paper the other missed is the whole argument for fusing them,
    # and the usual explanation for a result that looks wrong.
    text = explain(
        {"rrf_k": 60, "runs": {"bm25": {"rank": 1, "score": 9.63, "contribution": 0.0164}}}
    )
    assert "Only one retriever found this paper" in text


def test_degraded_search_is_reported_loudly():
    text = explain(
        {
            "rrf_k": 60,
            "runs": {"dense": {"rank": 2, "score": 0.8, "contribution": 0.0161}},
            "degraded": ["bm25"],
        }
    )
    assert "WARNING" in text
    assert "bm25" in text
    assert "do not reflect the full configured system" in text


def test_dense_only_result_reports_similarity():
    text = explain({"retriever": "dense", "chunk_index": 0, "similarity": 0.7123})
    assert "dense" in text
    assert "0.7123" in text


def test_bm25_only_result_reports_score():
    text = explain({"retriever": "bm25", "chunk_index": 2, "bm25_score": 9.6292})
    assert "bm25" in text
    assert "9.6292" in text


@pytest.mark.parametrize(
    "debug",
    [
        None,
        "a string",
        42,
        [],
        {},
        {"runs": {}},
        {"runs": "not a mapping"},
        {"runs": {"bm25": "not a mapping"}},
        {"runs": {"bm25": {"score": 9.6}}},  # no rank
        {"retriever": ""},
        {"unrecognised": "shape"},
    ],
)
def test_unreadable_payloads_fail_closed(debug):
    assert explain(debug) == _UNKNOWN


def test_boolean_is_not_treated_as_a_score():
    # bool is an int subclass. Without an explicit guard, True formats as
    # 1.0000 and reads as a genuine score rather than a type error.
    text = explain({"retriever": "dense", "similarity": True})
    assert "1.0000" not in text
    assert text == "Retrieved by dense."


def test_rank_without_score_still_explains():
    # Partial data is readable, so describe what is there rather than refusing.
    text = explain({"runs": {"dense": {"rank": 4}}})
    assert "dense ranked it #4" in text
