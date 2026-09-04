"""Is the difference between two modes real, or is it forty queries of noise?

**Why this module exists.** The ablation reports dense at nDCG@10 0.691 and
hybrid at 0.649. Presenting that as "dense is better" assumes the gap is larger
than what forty queries can produce by chance, and nothing in the metrics
module checks that assumption. Publishing a difference without a confidence
interval is the same category of mistake as publishing no numbers at all: it
looks like evidence and is not.

**Why a paired bootstrap rather than a t-test.** Per-query nDCG is bounded in
[0, 1], heavily clustered at 0 and 1, and nothing like normal - a t-test's
assumptions are simply false here. The bootstrap makes no distributional
assumption: it resamples the queries themselves, which is the thing that was
sampled in the first place.

**Paired, not independent.** Both modes are scored on the *same* forty queries,
so the comparison should resample query indices once and take both modes'
scores for the chosen queries. Resampling the two lists independently would
throw away the pairing and produce an interval far wider than the truth, since
most of the variance between queries is "some queries are hard" rather than
"some modes are better".

**What forty queries can and cannot support.** A 95% interval on a mean
difference from n=40 is wide. If it straddles zero the honest report is "no
detectable difference", not "they are the same" and certainly not "ours won by
a nose". This is the module that stops the ablation table from being read more
confidently than it deserves.
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass

# Enough resamples that the interval is stable to about three decimal places,
# and small enough to run inside a test.
DEFAULT_ITERATIONS = 10_000

# Fixed so the published interval is reproducible. A significance result that
# moves between runs invites re-rolling until it says something convenient.
DEFAULT_SEED = 20260910


@dataclass(frozen=True)
class Comparison:
    """The difference between two modes on the same queries."""

    n: int
    mean_a: float
    mean_b: float
    difference: float
    ci_low: float
    ci_high: float
    p_value: float
    wins_a: int
    wins_b: int
    ties: int

    @property
    def significant(self) -> bool:
        """True when the interval excludes zero.

        Stated as a property rather than left to the reader because "the
        interval contains zero but the difference looks big" is exactly the
        situation where a number gets quoted without its caveat.
        """
        return (self.ci_low > 0) or (self.ci_high < 0)

    def describe(self, name_a: str = "a", name_b: str = "b") -> str:
        gap = f"{self.difference:+.3f}"
        interval = f"[{self.ci_low:+.3f}, {self.ci_high:+.3f}]"
        verdict = "significant" if self.significant else "not distinguishable from zero"
        return (
            f"{name_b} - {name_a} = {gap} 95% CI {interval}, p={self.p_value:.3f} "
            f"({verdict}; {self.wins_b} wins / {self.wins_a} losses / {self.ties} ties "
            f"over n={self.n})"
        )


def compare(
    scores_a: Sequence[float],
    scores_b: Sequence[float],
    iterations: int = DEFAULT_ITERATIONS,
    seed: int = DEFAULT_SEED,
    confidence: float = 0.95,
) -> Comparison:
    """Paired bootstrap of ``mean(b) - mean(a)``.

    The p-value is a paired permutation-style two-sided test: under the null
    the two modes are interchangeable on any given query, so the sign of each
    per-query difference is a coin flip. Counting how often a random reflipping
    produces a mean difference at least as extreme as the observed one gives the
    probability of seeing this gap if the modes were equivalent.
    """
    if len(scores_a) != len(scores_b):
        raise ValueError(
            f"paired comparison needs the same queries on both sides, "
            f"got {len(scores_a)} and {len(scores_b)}"
        )
    if not scores_a:
        raise ValueError("nothing to compare")

    n = len(scores_a)
    diffs = [b - a for a, b in zip(scores_a, scores_b, strict=True)]
    observed = sum(diffs) / n

    rng = random.Random(seed)  # noqa: S311 - resampling, not cryptography

    # Confidence interval: resample queries with replacement, keeping pairs
    # together.
    resampled = []
    for _ in range(iterations):
        picks = [rng.randrange(n) for _ in range(n)]
        resampled.append(sum(diffs[i] for i in picks) / n)
    resampled.sort()

    tail = (1 - confidence) / 2
    low = resampled[int(tail * iterations)]
    high = resampled[min(int((1 - tail) * iterations), iterations - 1)]

    # p-value: flip the sign of each per-query difference at random.
    at_least_as_extreme = 0
    for _ in range(iterations):
        flipped = sum(diff if rng.random() < 0.5 else -diff for diff in diffs) / n
        if abs(flipped) >= abs(observed):
            at_least_as_extreme += 1
    # Add-one smoothing: with 10,000 resamples the smallest honest claim is
    # "p < 1e-4", and reporting exactly 0.0 would overstate it.
    p_value = (at_least_as_extreme + 1) / (iterations + 1)

    return Comparison(
        n=n,
        mean_a=sum(scores_a) / n,
        mean_b=sum(scores_b) / n,
        difference=observed,
        ci_low=low,
        ci_high=high,
        p_value=p_value,
        wins_a=sum(1 for diff in diffs if diff < 0),
        wins_b=sum(1 for diff in diffs if diff > 0),
        ties=sum(1 for diff in diffs if diff == 0),
    )


__all__ = ["DEFAULT_ITERATIONS", "DEFAULT_SEED", "Comparison", "compare"]
