# The golden set: how it was built, and what is wrong with it

Fifty queries, fifty relevance judgements, one person, one afternoon. This file
exists so that nobody - including me, later - can quote a number from this set
without knowing what it is worth.

## What it is

`v1.yaml` holds fifty queries against a snapshot of 3,377 arXiv papers, split
evenly across five classes:

| Class | What it tests | Expected winner |
|---|---|---|
| `exact_term` | Specialist vocabulary typed verbatim | Lexical (BM25) |
| `paraphrase` | The same need in everyday words | Dense |
| `conceptual` | A *why* or *how* question with no single answer document | Dense, weakly |
| `multi_hop` | Two topics that must **both** be satisfied | Hybrid, if anything |
| `unanswerable` | A plausible question the corpus cannot answer | Nothing should win |

The classes exist because a single average is a way of not looking. A system
can be excellent at jargon and useless at plain English and still post a
respectable mean. If the ablation cannot say *where* each retriever wins, it
has not explained anything, and the choice to run a hybrid is decoration.

The `unanswerable` class carries most of the weight. Every other class asks
whether the right thing is returned. This one asks whether the wrong thing is
withheld, which no ranking metric measures and which is the failure users
actually get burned by. On the day the set was written, all three retrieval
modes returned ten confident, well-formed, entirely irrelevant results for
every one of these ten queries. Precision@10 was zero and neither the API nor
the system had any way of noticing.

## How candidates were found

Judging 50 queries against 3,377 papers exhaustively is 168,850 decisions. That
was never going to happen, so this uses **pooling**, the method TREC has used
since 1992: run several retrievers, take the union of their top *k*, judge only
that union, and treat everything unjudged as irrelevant.

Here, the pool is the union of BM25 and dense retrieval at depth 12, produced
by `scripts/pool_candidates.py`. Both retrievers were used deliberately. Pooling
from one system and then evaluating that same system is circular: it can only
ever be marked on work it already did.

That script runs the retrievers in-process rather than through the HTTP API.
This is building the ruler, not using it - the API's throttling, serialisation
and ranking contract are irrelevant to curation and would only add failure
modes to a step that must not fail quietly.

## What is wrong with it

### 1. Pooling bias — the big one

A paper that neither retriever surfaced can never be marked relevant. So what
this set measures is **recall at pool depth 12**, not recall. True recall is
unknown and unknowable without reading all 3,377 abstracts fifty times.

This is not hypothetical here. It happened, repeatedly, and it is worth being
concrete about because the failure is so much less abstract than the name
suggests:

- **`para_10`** asks for "ranking documents when the reader's words differ from
  the author's words". Paper `2608.00452` opens by defining exactly that -
  *"when a relevant document phrases an answer differently from the query, BM25
  never retrieves it"* - and **neither retriever returned it**. The paper about
  the vocabulary gap was lost to the vocabulary gap.
- **`para_02`** ("improving a search engine by re-reading its own top results")
  is plain English for pseudo-relevance feedback. Its pool contained neither of
  the two PRF papers that rank in the top three for the jargon phrasing in
  `exact_03`.
- **`para_09`** ("how a dying massive star manages to explode outward") pooled
  essentially none of the core-collapse supernova papers that `exact_05`
  surfaced immediately.

Mitigation: **cross-pollination**. Where a paper was known to be relevant from
another query's pool, it was labelled here too, even though this query's pool
missed it. Affected queries are marked `CROSS-POLLINATED` in `v1.yaml`:
`para_02`, `para_04`, `para_09`, `para_10`, `conc_04`, `conc_06`, `conc_08`,
`conc_09`, `hop_06`, `hop_09`.

This makes the set harder and more honest, and it deliberately biases *against*
the systems being measured - which is the correct direction for a bias you
cannot remove. It does not eliminate the problem. Papers that no query's pool
ever surfaced remain invisible.

### 2. One judge, no agreement statistics

One person made every call. There is no second annotator, no adjudication of
disagreements, and therefore no inter-annotator agreement figure. Published IR
collections report Cohen's kappa or similar precisely because relevance is not
self-evident. This set has no such number, so treat differences between systems
of a few points as noise.

### 3. Judged from titles and 240 characters of abstract

Full texts were not read. For `exact_term` queries this is usually enough - the
technique is named in the title. For `conceptual` queries it is thin: a paper
can spend its introduction on exactly the question asked and never say so in
its first 240 characters.

### 4. Binary relevance, not graded

Every paper is relevant or it is not. Real relevance is graded, and several
calls were genuinely borderline. Two that were excluded and could reasonably
have been included at a lower grade:

- `2607.28381` (vector-like quarks decaying to two photons) for `exact_01` -
  a diphoton resonance search, but the particle is not framed as an ALP.
- `2606.11230` (TAMBO) for `exact_02` - mentions the look-elsewhere effect,
  but is a telescope proposal, not a paper about the correction.

Consequence: nDCG@10 is computed over binary gains, which is a weaker use of
nDCG than it was designed for. It still distinguishes *rank* - a relevant paper
at position 1 beats the same paper at position 8 - which is why it is worth
computing at all.

### 5. The author of the system wrote the labels

The same person built the retrievers and decided what counts as a correct
answer. That is a conflict of interest with no clean fix at this scale. The
partial defences: query classes and their expected winners were fixed *before*
any retrieval was run, so the shape of the result was committed to in advance;
judgements were made from a pooled list that does not identify which retriever
found what; and the whole file, including this section, ships in the repository
where anyone can disagree with a specific call.

### 6. Queries were written after seeing the corpus

Skimming category counts and sample titles first meant the queries are
answerable, which is necessary. It also means the corpus shaped the questions -
these are not the queries a stranger would type. Deliberate counterweights:

- `conc_02` ("why do neutrino experiments need such enormous detectors") and
  `hop_01` ("neutrino mass and cosmological structure formation") are marked
  `WEAK SUPPORT` and kept anyway. A set containing only well-supported queries
  would overstate how often a real question has a good answer.
- The ten `unanswerable` queries are *hard* negatives, not obvious ones. The
  corpus does contain medical, legal, financial and biomedical papers, so those
  topic areas are present and only the answers are missing. `none_04`
  ("diagnosing diabetic retinopathy from retinal photographs") sits beside real
  medical-imaging work. A system that abstains merely because a query looks
  unfamiliar has not solved anything.

### 7. The labels expire when the corpus moves

Relevance is a property of a query *and* a collection. `corpus:` in `v1.yaml`
records the snapshot: 3,377 papers, 3,465 chunks, captured 2026-09-04. Ingest
more papers and these labels become incomplete - newly arrived relevant papers
are unlabelled and will be scored as mistakes. Numbers from different corpus
sizes do not belong in the same table.

When the corpus changes materially, the honest move is a `v2.yaml`, not an edit
to this one.

## Using it

```python
from evaluation.golden import load_golden_set

golden = load_golden_set()  # validates on load, raises GoldenSetError
len(golden)  # 50
golden.by_class("unanswerable")  # 10 queries, all with empty relevant_ids
```

The loader validates hard, because every way this file can break is silent. The
sharpest example: `2607.07800` unquoted is valid YAML for the float `2607.078`.
It parses, it looks correct in a diff, and it matches no paper - so the query
scores zero for every retriever equally, which in an ablation table is
indistinguishable from a query that is simply hard. All identifiers are quoted
and a test asserts they are strings.
