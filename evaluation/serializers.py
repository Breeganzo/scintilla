"""Serialisers for evaluation runs.

These live in the app that owns the model. An earlier version of the API served
a second, unrelated ``EvaluationRun`` defined in ``papers``, which no harness
ever wrote to, so the endpoint returned an empty list while six real runs sat
in this app's table. Keeping the serialiser next to the model it serialises
makes that class of drift harder to reintroduce.
"""

from rest_framework import serializers

from evaluation.models import EvaluationRun


class EvaluationRunSerializer(serializers.ModelSerializer):
    """One measured sweep, with the provenance needed to reproduce it."""

    class Meta:
        model = EvaluationRun
        fields = [
            "id",
            "created_at",
            "mode",
            "top_k",
            "golden_set_version",
            "corpus_papers",
            "corpus_chunks",
            "git_sha",
            "metrics",
            "notes",
        ]
