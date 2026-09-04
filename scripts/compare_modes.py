"""Compare the three retrieval modes on the same queries, over HTTP.

Not a test. This exists to produce the informal observations the Day 7 plan
asks for, which become hypotheses that Day 9 checks against real metrics. It
goes through the HTTP endpoint deliberately: anything that bypassed the API
would be measuring a code path users never hit.

    python scripts/compare_modes.py [base_url]
"""

from __future__ import annotations

import json
import sys
import urllib.request

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8001"
TOP_K = 5

QUERIES = [
    ("exact_term", "Higgs boson coupling measurement"),
    ("paraphrase", "how do researchers rank documents when answering a search"),
    ("conceptual", "why is dark matter hard to detect directly"),
    ("multi_hop", "neutrino mass and cosmological structure formation"),
    ("unanswerable", "protein folding prediction with deep neural networks"),
]


def search(query: str, mode: str, top_k: int = TOP_K) -> dict:
    payload = json.dumps({"query": query, "mode": mode, "top_k": top_k}).encode()
    request = urllib.request.Request(  # noqa: S310 - BASE is a literal or argv, http only
        f"{BASE}/api/search/",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=60) as response:  # noqa: S310
        return json.loads(response.read())


def main() -> None:
    for label, query in QUERIES:
        print(f"\n{'=' * 78}\n{label.upper()}: {query}\n{'=' * 78}")

        tops: dict[str, list[str]] = {}
        for mode in ("bm25", "dense", "hybrid"):
            data = search(query, mode)
            results = data["results"]
            tops[mode] = [item["arxiv_id"] for item in results]

            print(f"\n  {mode:6s} ({data['took_ms']:.0f} ms)")
            for item in results[:3]:
                print(f"    {item['rank']}. {item['arxiv_id']}  {item['title'][:64]}")
            if not results:
                print("    (nothing)")

        lexical, dense = set(tops["bm25"][:3]), set(tops["dense"][:3])
        overlap = lexical & dense
        print(f"\n  top-3 overlap between bm25 and dense: {len(overlap)}/3 {sorted(overlap)}")
        print(f"  hybrid top-3: {tops['hybrid'][:3]}")

        # Which retriever's opinion did fusion actually adopt? If hybrid's top
        # hit came from only one list, that is the interesting case.
        detail = search(query, "hybrid", top_k=1)["results"]
        if detail:
            runs = detail[0]["debug"].get("runs", {})
            found_by = ", ".join(f"{name}@{info['rank']}" for name, info in sorted(runs.items()))
            print(f"  hybrid #1 was found by: {found_by}")


if __name__ == "__main__":
    main()
