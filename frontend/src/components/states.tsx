import type { ApiFailureKind } from "../api/client";
import { CORPUS } from "../lib/modes";

export function Skeleton() {
  return (
    <ul className="space-y-3" aria-busy="true" aria-label="Searching">
      {[0, 1, 2, 3].map((index) => (
        <li key={index} className="rounded-xl border border-line p-4">
          <div className="animate-pulse-bar space-y-2.5">
            <div className="h-3.5 w-3/4 rounded bg-line" />
            <div className="h-2.5 w-1/3 rounded bg-line" />
            <div className="h-2.5 w-full rounded bg-line" />
            <div className="h-2.5 w-5/6 rounded bg-line" />
          </div>
        </li>
      ))}
    </ul>
  );
}

export function Idle() {
  return (
    <div className="rounded-xl border border-dashed border-line px-5 py-8 text-center">
      <p className="text-[14px] text-muted">
        {CORPUS.papers.toLocaleString()} arXiv papers across{" "}
        <span className="font-mono text-[13px]">{CORPUS.categories.join(", ")}</span>.
      </p>
      <p className="mx-auto mt-2 max-w-lg text-[13px] leading-relaxed text-muted">
        Ask a question, then open <span className="font-medium text-ink">Why this result</span> on
        any paper to see which retriever found it and what that contributed to the ranking.
      </p>
    </div>
  );
}

export function Empty({ query }: { query: string }) {
  return (
    <div className="rounded-xl border border-line bg-surface px-5 py-8 text-center">
      <p className="text-[14px] font-medium">Nothing matched “{query}”.</p>
      <p className="mx-auto mt-2 max-w-lg text-[13px] leading-relaxed text-muted">
        The corpus is particle physics, cosmology and information retrieval only. A question
        outside it genuinely has no answer here — which is the honest outcome, and the one the
        evaluation deliberately tests for with ten unanswerable queries.
      </p>
    </div>
  );
}

const FAILURE_TITLE: Record<ApiFailureKind, string> = {
  network: "Could not reach the API",
  throttled: "Slow down a moment",
  invalid: "That query was rejected",
  server: "The API returned an error",
};

export function Failure({ kind, message }: { kind: ApiFailureKind; message: string }) {
  const amber = kind === "throttled";
  return (
    <div
      role="alert"
      className={[
        "rounded-xl border px-5 py-4",
        amber ? "border-amber-300 bg-amber-50 text-amber-900" : "border-red-300 bg-red-50 text-red-900",
      ].join(" ")}
    >
      <p className="text-[14px] font-medium">{FAILURE_TITLE[kind]}</p>
      <p className="mt-1 text-[13px] leading-relaxed">{message}</p>
      {kind === "throttled" && (
        <p className="mt-1.5 text-[12.5px]">
          Search is capped at 60 requests a minute on its own throttle scope, separate from the
          rest of the API — a dense query loads a model and scans an HNSW index, so it is the
          most expensive endpoint here.
        </p>
      )}
    </div>
  );
}
