"""Three tools: find papers, read one, and check how much to trust the finding.

The third tool is the unusual one. Retrieval quality is a measured property of
this system, and the measurement contradicts its own design - hybrid retrieval
is not significantly better than plain dense retrieval here. A calling model
that does not know that will over-trust a hybrid result set. Publishing the
numbers as a tool makes the caller's confidence answerable from evidence rather
than from the fact that a search returned ten rows.
"""

from typing import Annotated, Any, Literal

import httpx
from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from pydantic import Field

from mcp_server.client import Err, ScintillaClient
from mcp_server.config import Settings
from mcp_server.explain import explain

INSTRUCTIONS = """\
Scintilla is a retrieval system over a corpus of arXiv papers in high-energy \
physics, astrophysics and information retrieval. It answers nothing itself; it \
returns ranked papers and the evidence for why each one ranked where it did.

Prefer `mode="dense"`. It is the measured best on this corpus. Call \
`retrieval_report` before drawing strong conclusions from a result set: it \
reports the recorded quality of each mode, and those numbers are lower than \
people usually assume.

The corpus is finite and specialised. If a query is outside high-energy \
physics, astrophysics or information retrieval, the system will still return \
its ten closest matches - it does not abstain. Ten confident-looking results \
are not evidence that the corpus contains an answer.\
"""

# Written into the search tool's own description because a model reads the
# description before it chooses arguments. Putting the caveat in the response
# instead would be too late: the mode has already been picked by then.
_MODE_GUIDANCE = """\
Retrieval mode. Measured nDCG@10 over a 50-query hand-labelled golden set:
  dense  0.691  - the measured best; use this unless you have a reason not to
  hybrid 0.649  - NOT significantly better than dense (p = 0.194); slower
  bm25   0.473  - exact keyword matching; much weaker on paraphrased queries
  (recall@10 of 0.206 against dense's 0.615), but it sometimes finds papers
  dense misses entirely, which is why hybrid exists at all.\
"""


def _fail(prefix: str, err: Err) -> ToolError:
    """Turn a classified API failure into an error the caller can act on."""
    return ToolError(f"{prefix} ({err.kind}) {err.message}")


def build_server(
    settings: Settings | None = None,
    transport: httpx.AsyncBaseTransport | None = None,
) -> MCPServer:
    """Construct the server.

    ``transport`` is the injection seam the tests use to exercise the tools
    without opening a socket. In production it is None and httpx supplies its
    own.
    """
    resolved = settings or Settings.from_env()

    def open_client() -> ScintillaClient:
        return ScintillaClient(resolved, transport=transport)

    server = MCPServer(
        name="scintilla",
        title="Scintilla - measured retrieval over arXiv",
        instructions=INSTRUCTIONS,
        version="0.1.0",
    )

    @server.tool(
        name="search_papers",
        title="Search arXiv papers",
        description=(
            "Search the indexed arXiv corpus and return ranked papers, each "
            "with an explanation of why it ranked where it did. Returns "
            "metadata and ranking evidence, not paper text - call get_paper "
            "for the abstract."
        ),
    )
    async def search_papers(
        query: Annotated[
            str,
            Field(description=("A natural-language question or phrase. Maximum 512 characters.")),
        ],
        mode: Annotated[
            Literal["bm25", "dense", "hybrid"],
            Field(description=_MODE_GUIDANCE),
        ] = "dense",
        top_k: Annotated[
            int,
            Field(description="How many papers to return. Between 1 and 50."),
        ] = 10,
    ) -> dict[str, Any]:
        """Search the corpus and explain each result's position."""
        async with open_client() as client:
            result = await client.search(query, mode, top_k)

        if isinstance(result, Err):
            raise _fail("Search failed.", result)

        body: Any = result.value
        raw_results = body.get("results", []) if isinstance(body, dict) else []

        papers = [
            {
                "rank": item.get("rank"),
                "arxiv_id": item.get("arxiv_id"),
                "title": item.get("title"),
                "authors": item.get("authors", []),
                "primary_category": item.get("primary_category"),
                "published_at": item.get("published_at"),
                "url": item.get("abs_url"),
                "why_this_ranked_here": explain(item.get("debug")),
            }
            for item in raw_results
            if isinstance(item, dict)
        ]

        response: dict[str, Any] = {
            "query": body.get("query") if isinstance(body, dict) else query,
            "mode": body.get("mode") if isinstance(body, dict) else mode,
            "count": len(papers),
            "took_ms": body.get("took_ms") if isinstance(body, dict) else None,
            "results": papers,
        }

        diagnostics = body.get("diagnostics") if isinstance(body, dict) else None
        degraded = diagnostics.get("degraded") if isinstance(diagnostics, dict) else None
        if degraded:
            # Never let this be quiet. A degraded hybrid search silently becomes
            # a single-retriever search, and every quality number the caller
            # might reasonably assume then describes a different system.
            response["warning"] = (
                f"Degraded: {', '.join(str(d) for d in degraded)} unavailable. "
                "These results do not reflect the full configured system."
            )
        return response

    @server.tool(
        name="get_paper",
        title="Read one paper's record",
        description=(
            "Retrieve the full stored record for one arXiv paper: title, "
            "authors, abstract, categories and publication date. Use the "
            "arxiv_id returned by search_papers."
        ),
    )
    async def get_paper(
        arxiv_id: Annotated[
            str,
            Field(
                description=(
                    'The arXiv identifier, for example "2606.20830". Use an id '
                    "returned by search_papers."
                )
            ),
        ],
    ) -> dict[str, Any]:
        """Fetch one paper's full stored record."""
        async with open_client() as client:
            result = await client.get_paper(arxiv_id)

        if isinstance(result, Err):
            if result.kind == "not_found":
                raise ToolError(
                    f"No paper with arXiv id {arxiv_id!r} is in this corpus. The "
                    "corpus is a fixed subset of arXiv, so a valid arXiv id may "
                    "still be absent."
                )
            raise _fail(f"Could not read paper {arxiv_id!r}.", result)
        return result.value

    @server.tool(
        name="retrieval_report",
        title="How well does this retrieval actually work",
        description=(
            "Report the measured retrieval quality of each mode against a "
            "hand-labelled golden set, with the commit and golden-set version "
            "the numbers came from. Call this before treating a result set as "
            "authoritative."
        ),
    )
    async def retrieval_report(
        mode: Annotated[
            Literal["bm25", "dense", "hybrid"] | None,
            Field(description=("Restrict the report to one retrieval mode. Omit for all three.")),
        ] = None,
    ) -> dict[str, Any]:
        """Report the measured retrieval quality recorded for this deployment."""
        async with open_client() as client:
            result = await client.evaluation_runs(mode)

        if isinstance(result, Err):
            raise _fail("Could not read evaluation history.", result)

        runs: list[Any] = result.value
        if not runs:
            # Distinct from "quality is zero", and the difference matters. This
            # deployment simply has no recorded measurement.
            return {
                "runs": [],
                "note": (
                    "No evaluation runs are recorded in this deployment. Retrieval "
                    "quality here is unmeasured - treat results accordingly."
                ),
            }

        latest_by_mode: dict[str, dict[str, Any]] = {}
        for run in runs:
            if not isinstance(run, dict):
                continue
            run_mode = str(run.get("mode", ""))
            # Runs arrive newest first, so the first of each mode wins.
            if run_mode and run_mode not in latest_by_mode:
                latest_by_mode[run_mode] = {
                    "mode": run_mode,
                    "measured_at": run.get("created_at"),
                    "git_sha": run.get("git_sha"),
                    "golden_set_version": run.get("golden_set_version"),
                    "top_k": run.get("top_k"),
                    # The corpus is part of the measurement. Labels were assigned
                    # against a snapshot, so a number from a differently-sized
                    # corpus is not comparable to this one.
                    "corpus_papers": run.get("corpus_papers"),
                    "metrics": run.get("metrics"),
                }

        return {
            "runs": list(latest_by_mode.values()),
            "how_to_read_this": (
                "Macro-averaged over hand-labelled queries; unanswerable queries "
                "are held out. Dense beats BM25 by a wide and statistically "
                "significant margin. Hybrid does NOT significantly beat dense "
                "(difference -0.041 nDCG@10, 95% CI [-0.103, +0.020], p = 0.194) "
                "- the honest reading is that they are indistinguishable on this "
                "corpus and the simpler system is preferable. Metrics are tied to "
                "a git_sha and a golden_set_version; comparing numbers across "
                "different golden-set versions is invalid."
            ),
        }

    return server
