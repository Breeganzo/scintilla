import { useCallback, useRef, useState } from "react";

import type { ApiFailureKind, SearchMode, SearchResponse } from "./api/client";
import { search } from "./api/client";
import { DiagnosticsBar } from "./components/DiagnosticsBar";
import { ModeToggle } from "./components/ModeToggle";
import { ResultCard } from "./components/ResultCard";
import { SearchBar } from "./components/SearchBar";
import { Empty, Failure, Idle, Skeleton } from "./components/states";
import { CORPUS, DEFAULT_MODE, DEFAULT_TOP_K, REPO_URL } from "./lib/modes";

type Phase =
  | { status: "idle" }
  | { status: "loading" }
  | { status: "loaded"; response: SearchResponse }
  | { status: "failed"; kind: ApiFailureKind; message: string };

export default function App() {
  const [mode, setMode] = useState<SearchMode>(DEFAULT_MODE);
  const [query, setQuery] = useState("");
  const [phase, setPhase] = useState<Phase>({ status: "idle" });

  // One in-flight request at a time. Without this, switching mode twice quickly
  // leaves two dense searches racing and the slower one can land last - so the
  // results on screen would belong to a mode the toggle no longer shows. In an
  // application whose entire claim is about ranking, that bug reads as "the
  // ranking is wrong", which is the most expensive possible misdiagnosis.
  const inFlight = useRef<AbortController | null>(null);

  const run = useCallback(async (nextQuery: string, nextMode: SearchMode) => {
    inFlight.current?.abort();
    const controller = new AbortController();
    inFlight.current = controller;

    setQuery(nextQuery);
    setPhase({ status: "loading" });

    try {
      const result = await search({
        query: nextQuery,
        mode: nextMode,
        topK: DEFAULT_TOP_K,
        signal: controller.signal,
      });
      if (controller.signal.aborted) return;

      setPhase(
        result.ok
          ? { status: "loaded", response: result.data }
          : { status: "failed", kind: result.kind, message: result.message },
      );
    } catch {
      // Only an abort reaches here; the client turns every other failure into a
      // value. A superseded request must leave the newer one's state alone.
    } finally {
      if (inFlight.current === controller) inFlight.current = null;
    }
  }, []);

  const onModeChange = (nextMode: SearchMode) => {
    setMode(nextMode);
    // Re-running immediately is what makes the toggle an ablation you can feel
    // rather than a setting you have to remember to re-apply.
    if (query) void run(query, nextMode);
  };

  return (
    <div className="mx-auto min-h-dvh max-w-3xl px-5 py-10">
      <header className="flex items-start justify-between gap-4">
        <div>
          <h1 className="text-[22px] font-semibold tracking-tight">Scintilla</h1>
          <p className="mt-0.5 text-[13.5px] text-muted">
            Retrieval over arXiv — and the evaluation that says which strategy actually works.
          </p>
        </div>
        <a
          href={REPO_URL}
          target="_blank"
          rel="noopener noreferrer"
          className="shrink-0 rounded-lg border border-line px-2.5 py-1.5 text-[12.5px] text-muted transition-colors hover:border-accent hover:text-accent"
        >
          Source ↗
        </a>
      </header>

      <main className="mt-7 space-y-5">
        <SearchBar
          initialQuery=""
          busy={phase.status === "loading"}
          onSearch={(next) => void run(next, mode)}
        />

        <ModeToggle mode={mode} onChange={onModeChange} />

        <section aria-live="polite" aria-busy={phase.status === "loading"} className="space-y-3">
          {phase.status === "idle" && <Idle />}
          {phase.status === "loading" && <Skeleton />}
          {phase.status === "failed" && <Failure kind={phase.kind} message={phase.message} />}
          {phase.status === "loaded" && (
            <>
              <DiagnosticsBar response={phase.response} />
              {phase.response.results.length === 0 ? (
                <Empty query={phase.response.query} />
              ) : (
                <ul className="space-y-3">
                  {phase.response.results.map((result) => (
                    <ResultCard key={result.arxiv_id} result={result} />
                  ))}
                </ul>
              )}
            </>
          )}
        </section>
      </main>

      <footer className="mt-12 border-t border-line pt-4 text-[12px] leading-relaxed text-muted">
        <p>
          {CORPUS.papers.toLocaleString()} papers · {CORPUS.chunks.toLocaleString()} chunks ·
          BGE-small-en-v1.5, 384 dimensions · pgvector HNSW for dense, OpenSearch BM25 for
          lexical.
        </p>
        <p className="mt-1">
          Quality figures come from a 50-query golden set labelled by hand, not from a benchmark
          this system was tuned against. Answer generation is measured but not yet served here.
        </p>
      </footer>
    </div>
  );
}
