import type { SearchResponse } from "../api/client";

interface Props {
  response: SearchResponse;
}

function readDegraded(diagnostics: Record<string, unknown>): string[] {
  const value = diagnostics["degraded"];
  if (!Array.isArray(value)) return [];
  return value.filter((item): item is string => typeof item === "string");
}

function readNumber(diagnostics: Record<string, unknown>, key: string): number | null {
  const value = diagnostics[key];
  return typeof value === "number" ? value : null;
}

/**
 * What the request actually did.
 *
 * The degraded banner is the part that matters. When one retriever fails,
 * hybrid keeps working by silently becoming dense-only - which is the right
 * runtime behaviour and a terrible thing to leave invisible, because every
 * number on the page then describes a different system than the label says.
 * This project has already shipped two defects whose whole character was
 * "it kept working and stopped telling anyone". Not a third.
 */
export function DiagnosticsBar({ response }: Props) {
  const diagnostics = response.diagnostics as Record<string, unknown>;
  const degraded = readDegraded(diagnostics);
  const rrfK = readNumber(diagnostics, "rrf_k");
  const depth = readNumber(diagnostics, "fusion_depth");

  return (
    <div className="space-y-2">
      {degraded.length > 0 && (
        <div
          role="alert"
          className="rounded-lg border border-amber-300 bg-amber-50 px-3.5 py-2.5 text-[13px] leading-relaxed text-amber-900"
        >
          <span className="font-semibold">Degraded.</span> {degraded.join(" and ")} failed on this
          request, so these results come from the remaining retriever only. They are not a
          measurement of {response.mode} search, and should not be read as one.
        </div>
      )}

      <div className="flex flex-wrap items-center gap-x-3 gap-y-1 font-mono text-[12px] text-muted">
        <span className="font-sans font-medium text-ink">
          {response.count} {response.count === 1 ? "result" : "results"}
        </span>
        <span aria-hidden>·</span>
        <span title="Server-side, retrieval plus hydration. Not the in-process figure quoted on the mode cards.">
          {response.took_ms.toFixed(0)} ms end-to-end
        </span>
        <span aria-hidden>·</span>
        <span>{response.mode}</span>
        {rrfK !== null && (
          <>
            <span aria-hidden>·</span>
            <span>RRF k={rrfK}</span>
          </>
        )}
        {depth !== null && (
          <>
            <span aria-hidden>·</span>
            <span>fusion depth {depth}</span>
          </>
        )}
        <span aria-hidden>·</span>
        <span>top_k {response.top_k}</span>
      </div>
    </div>
  );
}
