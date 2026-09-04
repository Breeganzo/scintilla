"""Turning a paper into the text that actually gets embedded.

**The decision: one chunk per paper, ``title + "\\n\\n" + abstract``.**

The reasoning matters more than the code, so it is written down here rather
than left implicit.

A previous production system this design draws on split long documents
recursively at 1000 characters with 200 characters of overlap. That is a sound
default for reports and manuals, and it is the wrong choice here. An arXiv
abstract is 150-250 words that a physicist wrote to be read as one piece. The
opening sentence establishes what was measured and why; the closing sentence
gives the result. Split them apart and each fragment retrieves worse than the
whole: the first half has no numbers, the second half has no subject.

The title is prepended rather than dropped because it carries the highest-signal
terms in the record - detector names, particles, collaboration names - which
BM25 weights heavily and which the abstract sometimes never repeats.

The exception is handled rather than ignored. A small number of abstracts run
long enough to exceed the embedding model's context window. The model would
silently discard the tail, and silent truncation is the worst possible failure
mode: retrieval quality drops for a subset of documents with nothing in any log
to explain it. Those papers are sentence-split with one sentence of overlap,
the split is logged, and the count is recorded on the ingestion run.
"""

from __future__ import annotations

import logging
import math
import re
from collections.abc import Callable
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# BAAI/bge-small-en-v1.5 truncates at 512 tokens. Chunking to the same ceiling
# means nothing is ever handed to the model that it would silently cut.
DEFAULT_MAX_TOKENS = 512

# Split after . ! or ? when followed by whitespace and a capital or a digit.
# Deliberately simple. It mis-handles "et al." and "Fig. 3", which costs an
# occasional split in the wrong place inside an abstract already long enough to
# need splitting at all - a rare case within a rare case. A real sentence
# segmenter is a dependency and a model download for no measurable gain.
_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")

# Whitespace tokens under-count against a subword tokenizer, which splits
# "photoproduction" or "\\sqrt{s}" into several pieces.
#
# This constant was 1.3 and was described as a deliberate over-estimate. It was
# not. Measured against BGE-small's own tokenizer over 2,975 real arXiv
# abstracts: mean 1.63 real tokens per whitespace word, median 1.55, p99 2.96,
# max 5.77 (LaTeX-dense theory abstracts). At 1.3 the estimate came in *under*
# the truth for 94.2% of papers, and 69 abstracts that genuinely exceeded the
# 512-token ceiling were estimated at under it - so they were handed to the
# model whole and silently truncated. Exactly the failure the comment claimed
# to prevent.
#
# 1.7 sits above the median and the mean. It still under-counts the LaTeX-heavy
# tail, which is why this is now only a FALLBACK: callers that can reach the
# real tokenizer should pass ``count_tokens`` to chunk_paper and get an exact
# answer. The fallback exists so that chunking, and its tests, do not depend on
# PyTorch.
_TOKENS_PER_WORD = 1.7

# A function that returns the number of model tokens in a string.
TokenCounter = Callable[[str], int]


@dataclass(frozen=True, slots=True)
class TextChunk:
    """One unit of text destined for the embedding model."""

    index: int
    text: str
    token_count: int


def estimate_tokens(text: str) -> int:
    """Approximate the subword token count of ``text``."""
    return math.ceil(len(text.split()) * _TOKENS_PER_WORD)


def build_document_text(title: str, abstract: str) -> str:
    """Join title and abstract into the single string that gets embedded."""
    return f"{' '.join(title.split())}\n\n{' '.join(abstract.split())}"


def split_sentences(text: str) -> list[str]:
    """Split into sentences, preserving order and dropping empties."""
    return [sentence for sentence in _SENTENCE_BOUNDARY.split(text) if sentence.strip()]


def chunk_paper(
    title: str,
    abstract: str,
    *,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    overlap_sentences: int = 1,
    arxiv_id: str = "",
    count_tokens: TokenCounter | None = None,
) -> list[TextChunk]:
    """Produce the chunks for one paper.

    Returns exactly one chunk in the overwhelming majority of cases. More than
    one means the abstract exceeded ``max_tokens`` and was sentence-split, which
    is logged at WARNING - the caller is expected to count these.

    ``count_tokens`` should be the embedding model's own tokenizer. Chunking
    decides what the model sees, so counting with a different tokenizer than the
    model uses means the ceiling is enforced against the wrong number and long
    inputs are truncated without anything being logged. It defaults to the
    whitespace heuristic only so this module stays importable without PyTorch.
    """
    measure = count_tokens or estimate_tokens
    document = build_document_text(title, abstract)
    total_tokens = measure(document)

    if total_tokens <= max_tokens:
        return [TextChunk(index=0, text=document, token_count=total_tokens)]

    logger.warning(
        "Abstract for %s estimates %s tokens, over the %s ceiling; sentence-splitting",
        arxiv_id or "<unknown>",
        total_tokens,
        max_tokens,
    )

    sentences = split_sentences(document)
    if len(sentences) <= 1:
        # One enormous unsplittable sentence. Nothing sensible to do beyond
        # emitting it whole and being loud about it, because this is precisely
        # the case where the model will truncate.
        logger.warning(
            "Text for %s exceeds %s tokens but has no sentence boundary; "
            "the embedding model will truncate it",
            arxiv_id or "<unknown>",
            max_tokens,
        )
        return [TextChunk(index=0, text=document, token_count=total_tokens)]

    chunks: list[TextChunk] = []
    current: list[str] = []

    for sentence in sentences:
        candidate = [*current, sentence]
        if current and measure(" ".join(candidate)) > max_tokens:
            text = " ".join(current)
            chunks.append(TextChunk(index=len(chunks), text=text, token_count=measure(text)))
            # Carry the tail sentences forward so a claim is never separated
            # from the sentence that set it up.
            current = current[-overlap_sentences:] if overlap_sentences else []
            current.append(sentence)
        else:
            current = candidate

    if current:
        text = " ".join(current)
        chunks.append(TextChunk(index=len(chunks), text=text, token_count=measure(text)))

    return chunks
