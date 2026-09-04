"""Turn a result's ``debug`` payload into one sentence a model can reason about.

``debug`` is a DRF ``DictField``, so the published schema types it as an open
object - which is true, and means there is no contract to rely on. It is
therefore narrowed here at runtime, and the narrowing **fails closed**: an
unrecognised shape produces "no ranking explanation available" rather than a
confident sentence assembled from missing keys.

That bias is deliberate. This text exists so a caller can judge how much to
trust a hit. An explanation that is quietly wrong is worse than none at all,
because it is indistinguishable from one that is right.
"""

from typing import Any

# Reciprocal rank fusion contributes 1 / (k + rank) per retriever that found a
# document. Reported so the arithmetic can be checked rather than believed.
_UNKNOWN = "No ranking explanation available for this result."


def _as_number(value: Any) -> float | None:
    # bool is an int subclass, and True would format as 1.0000 - which looks
    # like a score rather than a type error.
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _describe_run(name: str, run: Any) -> str | None:
    if not isinstance(run, dict):
        return None
    rank = _as_number(run.get("rank"))
    score = _as_number(run.get("score"))
    contribution = _as_number(run.get("contribution"))
    if rank is None:
        return None

    parts = [f"{name} ranked it #{rank:g}"]
    if score is not None:
        parts.append(f"score {score:.4f}")
    if contribution is not None:
        parts.append(f"contributing {contribution:.5f}")
    return ", ".join(parts)


def explain(debug: Any) -> str:
    """Describe why a paper is at its position, or say that we cannot."""
    if not isinstance(debug, dict):
        return _UNKNOWN

    runs = debug.get("runs")
    if isinstance(runs, dict) and runs:
        described = [
            text
            for name, run in runs.items()
            if (text := _describe_run(str(name), run)) is not None
        ]
        if not described:
            return _UNKNOWN

        sentence = "Fused from " + "; ".join(described) + "."

        rrf_k = _as_number(debug.get("rrf_k"))
        if rrf_k is not None:
            sentence += f" Reciprocal rank fusion with k={rrf_k:g}."

        if len(described) == 1:
            # The single most useful thing this panel can say. One retriever
            # finding a paper the other missed is the entire argument for
            # fusing them, and also the usual explanation for a result that
            # looks wrong.
            sentence += " Only one retriever found this paper; the other missed it."

        degraded = debug.get("degraded")
        if isinstance(degraded, list) and degraded:
            names = ", ".join(str(d) for d in degraded)
            sentence += (
                f" WARNING: {names} failed during this search, so these results"
                " do not reflect the full configured system."
            )
        return sentence

    retriever = debug.get("retriever")
    if isinstance(retriever, str) and retriever:
        similarity = _as_number(debug.get("similarity"))
        bm25_score = _as_number(debug.get("bm25_score"))
        if similarity is not None:
            return f"Retrieved by {retriever}, cosine similarity {similarity:.4f}."
        if bm25_score is not None:
            return f"Retrieved by {retriever}, BM25 score {bm25_score:.4f}."
        return f"Retrieved by {retriever}."

    # Reached when the payload is a dict but matches neither known shape,
    # which is what a serializer change would look like from here.
    return _UNKNOWN
