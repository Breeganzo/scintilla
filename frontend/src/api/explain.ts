/**
 * Turning the untyped `debug` blob into something the UI can render.
 *
 * `debug` is a `DictField` on the backend, so the generated schema types it as
 * `Record<string, unknown>` - which is *correct*. The OpenAPI document genuinely
 * does not describe its contents, and pretending otherwise with a hand-written
 * interface would be inventing a contract the server never agreed to.
 *
 * So it is narrowed here, once, at runtime, with guards that fail closed. A
 * shape we do not recognise renders as "no explanation available" rather than
 * as a plausible-looking panel built from undefined values. An explanation that
 * is quietly wrong is worse than no explanation, because the entire point of
 * this panel is to be the thing you trust when a ranking looks odd.
 */

/** One retriever's contribution to a fused result. */
export interface RunContribution {
  retriever: string;
  rank: number;
  score: number;
  contribution: number;
}

export type Explanation =
  | {
      kind: "fused";
      rrfK: number;
      runs: RunContribution[];
      /** Retrievers that failed on this request, if any. */
      degraded: string[];
    }
  | {
      kind: "single";
      retriever: string;
      score: number;
      /** Which chunk of the paper matched. >0 means a split abstract. */
      chunkIndex: number | null;
      embeddingModel: string | null;
    }
  | { kind: "unknown" };

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function asNumber(value: unknown): number | null {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

function asStringArray(value: unknown): string[] {
  if (!Array.isArray(value)) return [];
  return value.filter((item): item is string => typeof item === "string");
}

export function explain(debug: Record<string, unknown>): Explanation {
  const rrfK = asNumber(debug["rrf_k"]);
  const runsRaw = debug["runs"];

  if (rrfK !== null && isRecord(runsRaw)) {
    const runs: RunContribution[] = [];
    for (const [retriever, detail] of Object.entries(runsRaw)) {
      if (!isRecord(detail)) continue;
      const rank = asNumber(detail["rank"]);
      const score = asNumber(detail["score"]);
      const contribution = asNumber(detail["contribution"]);
      if (rank === null || score === null || contribution === null) continue;
      runs.push({ retriever, rank, score, contribution });
    }
    if (runs.length === 0) return { kind: "unknown" };

    // Largest contribution first: the retriever that actually decided this
    // ranking should be the first thing read, not an alphabetical accident.
    runs.sort((a, b) => b.contribution - a.contribution);
    return {
      kind: "fused",
      rrfK,
      runs,
      degraded: asStringArray(debug["degraded"]),
    };
  }

  const retriever = debug["retriever"];
  if (typeof retriever === "string") {
    // bm25 reports `bm25_score`, dense reports `similarity`. Neither is a
    // probability and they are not comparable to each other, which is exactly
    // why RRF fuses ranks rather than these numbers.
    const score = asNumber(debug["bm25_score"]) ?? asNumber(debug["similarity"]);
    if (score === null) return { kind: "unknown" };
    const embeddingModel = debug["embedding_model"];
    return {
      kind: "single",
      retriever,
      score,
      chunkIndex: asNumber(debug["chunk_index"]),
      embeddingModel: typeof embeddingModel === "string" ? embeddingModel : null,
    };
  }

  return { kind: "unknown" };
}

/** Retrievers that were asked but returned nothing for this paper. */
export function missingFrom(explanation: Explanation, allRetrievers: string[]): string[] {
  if (explanation.kind !== "fused") return [];
  const found = new Set(explanation.runs.map((run) => run.retriever));
  return allRetrievers.filter((name) => !found.has(name));
}
