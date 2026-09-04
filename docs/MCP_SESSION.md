# An MCP session, verbatim

Captured against the live system: 3,377 papers, PostgreSQL 16 with pgvector,
OpenSearch 3.8. The client is the reference MCP SDK speaking stdio to
`python -m mcp_server`, which speaks HTTP to the Django API.

Nothing below is edited except for trimming the per-class metric breakdown in
the first response, which is long and repeats the same structure.

Reproduce it with the API running on `:8001`:

```bash
python -m mcp_server
```

---

## `retrieval_report(mode="dense")`

The first call a client should make. It answers "how good is this system,
actually" with numbers that are tied to the commit and golden set that produced
them.

```json
{
  "runs": [
    {
      "mode": "dense",
      "measured_at": "2026-09-04T11:37:07.595444Z",
      "git_sha": "010720aac5ad437d8cfe078ccbde665dce4573df",
      "golden_set_version": 1,
      "top_k": 10,
      "corpus_papers": 3377,
      "metrics": {
        "overall": {
          "mrr": 0.7747916666666667,
          "ndcg@1": 0.675,
          "ndcg@5": 0.638020258763893,
          "ndcg@10": 0.6909485782669981,
          "recall@1": 0.16614087301587302,
          "recall@5": 0.5629861111111112,
          "recall@10": 0.7440873015873015,
          "precision@1": 0.675,
          "precision@5": 0.53,
          "precision@10": 0.35750000000000004
        }
      }
    }
  ],
  "how_to_read_this": "Macro-averaged over hand-labelled queries; unanswerable queries are held out. Dense beats BM25 by a wide and statistically significant margin. Hybrid does NOT significantly beat dense (difference -0.041 nDCG@10, 95% CI [-0.103, +0.020], p = 0.194) - the honest reading is that they are indistinguishable on this corpus and the simpler system is preferable. Metrics are tied to a git_sha and a golden_set_version; comparing numbers across different golden-set versions is invalid."
}
```

Three things this makes hard to misread. The numbers arrive with
`git_sha`, `golden_set_version` and `corpus_papers`, so they cannot be quoted
without their provenance. `how_to_read_this` states that hybrid is **not**
significantly better than dense, which is the finding most likely to be assumed
away. And `recall@10` of 0.744 is a real ceiling: roughly a quarter of the
relevant material is not in the top ten.

---

## `search_papers(mode="hybrid")`

The showcase query. Every result explains its own position.

```json
{
  "query": "how bright was the beam when the collisions were recorded",
  "mode": "hybrid",
  "count": 2,
  "took_ms": 164.43,
  "results": [
    {
      "rank": 1,
      "arxiv_id": "2608.10988",
      "title": "High-energy electron-positron beam collisions with large-angle disruptions",
      "authors": [
        "W. Zhang",
        "T. Grismayer",
        "L. O. Silva"
      ],
      "primary_category": "physics.acc-ph",
      "published_at": "2026-08-11T14:43:50Z",
      "url": "https://arxiv.org/abs/2608.10988v1",
      "why_this_ranked_here": "Fused from bm25 ranked it #5, score 7.7659, contributing 0.01538; dense ranked it #3, score 0.6877, contributing 0.01587. Reciprocal rank fusion with k=60."
    },
    {
      "rank": 2,
      "arxiv_id": "2609.03452",
      "title": "Gamma-Ray Bursts and Optical Brightness: Insights from Swift Satellite",
      "authors": [
        "Istvan I. Racz",
        "Yasmin Nehme",
        "Lajos G. Balazs"
      ],
      "primary_category": "astro-ph.HE",
      "published_at": "2026-09-03T07:06:53Z",
      "url": "https://arxiv.org/abs/2609.03452v1",
      "why_this_ranked_here": "Fused from bm25 ranked it #6, score 6.4568, contributing 0.01515; dense ranked it #8, score 0.6774, contributing 0.01471. Reciprocal rank fusion with k=60."
    }
  ]
}
```

`why_this_ranked_here` is the same prose the browser shows, from the same
`debug` payload. The arithmetic is checkable: rank 5 in BM25 gives
`1 / (60 + 5) = 0.01538`, rank 3 in dense gives `1 / (60 + 3) = 0.01587`.

---

## A query the corpus cannot answer

This is the failure mode the server's instructions warn about, reproduced
deliberately.

```json
{
  "query": "what is the best recipe for sourdough bread",
  "mode": "dense",
  "count": 2,
  "took_ms": 134.51,
  "results": [
    {
      "rank": 1,
      "arxiv_id": "2608.29249",
      "title": "Validating FKG.in: Soundness Assessment in LLM-Augmented Indian Food Knowledge",
      "authors": [
        "Saransh Kumar Gupta",
        "Armaan Shah",
        "Lipika Dey",
        "Partha Pratim Das",
        "Ramesh Jain"
      ],
      "primary_category": "cs.AI",
      "published_at": "2026-08-29T13:10:06Z",
      "url": "https://arxiv.org/abs/2608.29249v2",
      "why_this_ranked_here": "Retrieved by dense, cosine similarity 0.5502."
    },
    {
      "rank": 2,
      "arxiv_id": "2607.23273",
      "title": "Statistically Supported LLM Ingredient and Recipe Data Collection in Computational Nutrition",
      "authors": [
        "James Izzard",
        "Hassan Eshkiki",
        "Fabio Caraffini"
      ],
      "primary_category": "cs.IR",
      "published_at": "2026-07-25T16:19:02Z",
      "url": "https://arxiv.org/abs/2607.23273v1",
      "why_this_ranked_here": "Retrieved by dense, cosine similarity 0.5489."
    }
  ]
}
```

There is no sourdough in a corpus of high-energy physics, astrophysics and
information retrieval. The system returns two papers anyway, at cosine
similarity 0.55, and they are *topically adjacent enough to look plausible* —
food knowledge graphs and recipe data collection. A model handed this without
warning would summarise them as if they were an answer.

**The retriever does not abstain.** It returns the nearest neighbours it has,
always. That is why the server's instructions say so before any tool is called,
and why the abstention logic measured in the answering layer sits above
retrieval rather than inside it.

---

## `get_paper` on an id that is not in the corpus

```
is_error: True
Error executing tool get_paper: No paper with arXiv id '9999.99999' is in this corpus. The corpus is a fixed subset of arXiv, so a valid arXiv id may still be absent.
```

A bare 404 would invite the conclusion that the paper does not exist. The error
says what is actually true: the corpus is a subset, and a perfectly valid arXiv
id can be absent from it.

---

## What this session found

`retrieval_report` returned an empty list the first time it was called.

Two `EvaluationRun` models existed — one scaffolded in Phase 1 under `papers`,
one under `evaluation` that the evaluation harness writes to. The API served
the first. Six real runs sat in the second, unreachable over HTTP.

The MCP server was the first consumer to *ask the API* for those numbers rather
than render a copy of them, which is why it surfaced here and not in the
browser or the test suite. See [ARCHITECTURE.md §7e](ARCHITECTURE.md) for the
fix and the reason the tests missed it.
