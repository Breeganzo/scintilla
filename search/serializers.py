"""Request and response shapes for the search endpoint.

The response is serialised explicitly rather than assembled as a dict, so that
``drf-spectacular`` can describe it in the OpenAPI schema. Day 11 generates the
frontend's TypeScript types from that schema, which turns a serializer change
that breaks the client into a build error rather than a runtime one.
"""

from __future__ import annotations

from rest_framework import serializers

from search.retrievers import DEFAULT_MODE, DEFAULT_TOP_K, MODES

# Long enough for a real research question, short enough that the embedding
# model is not handed something it will truncate anyway. Unbounded input here
# would be a cheap way to make every request pay for a full-length transformer
# forward pass.
MAX_QUERY_LENGTH = 512

MAX_TOP_K = 50


class SearchRequestSerializer(serializers.Serializer):
    """What a client may ask for."""

    query = serializers.CharField(
        max_length=MAX_QUERY_LENGTH,
        trim_whitespace=True,
        help_text="Natural-language query.",
    )
    mode = serializers.ChoiceField(
        choices=[(mode, mode) for mode in MODES],
        default=DEFAULT_MODE,
        help_text=(
            "Retrieval strategy. Exposed because it is the ablation mechanism: "
            "the evaluation harness calls this endpoint once per mode, so the "
            "measured system is the served system."
        ),
    )
    top_k = serializers.IntegerField(
        default=DEFAULT_TOP_K,
        min_value=1,
        # Capped because top_k sizes a database query and a response body.
        # An uncapped value is a denial-of-service vector dressed up as a
        # pagination parameter.
        max_value=MAX_TOP_K,
        help_text="How many results to return.",
    )

    def validate_query(self, value: str) -> str:
        if not value.strip():
            raise serializers.ValidationError("Query must not be blank.")
        return value


class SearchResultSerializer(serializers.Serializer):
    """One result, with the paper metadata needed to render it."""

    arxiv_id = serializers.CharField()
    rank = serializers.IntegerField()
    score = serializers.FloatField()
    title = serializers.CharField()
    abstract = serializers.CharField()
    authors = serializers.ListField(child=serializers.CharField())
    primary_category = serializers.CharField()
    published_at = serializers.DateTimeField()
    abs_url = serializers.URLField()
    debug = serializers.DictField(
        help_text=(
            "Per-retriever rank and score detail. For hybrid, this shows which "
            "retrievers found the paper and what each contributed to its fused "
            "score - the data behind the 'why this result' panel, and the first "
            "thing to read when a ranking looks wrong."
        )
    )


class SearchResponseSerializer(serializers.Serializer):
    """The envelope."""

    query = serializers.CharField()
    mode = serializers.CharField()
    top_k = serializers.IntegerField()
    count = serializers.IntegerField()
    took_ms = serializers.FloatField()
    diagnostics = serializers.DictField()
    results = SearchResultSerializer(many=True)


__all__ = [
    "MAX_QUERY_LENGTH",
    "MAX_TOP_K",
    "SearchRequestSerializer",
    "SearchResponseSerializer",
    "SearchResultSerializer",
]
