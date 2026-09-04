import type { Explanation } from "../api/explain";
import { missingFrom } from "../api/explain";

interface Props {
  explanation: Explanation;
  fusedScore: number;
}

const ALL_RETRIEVERS = ["bm25", "dense"];

const RETRIEVER_LABEL: Record<string, string> = {
  bm25: "BM25 (keyword)",
  dense: "Dense (meaning)",
};

/**
 * The panel this whole application exists to show.
 *
 * A search result with a score next to it is unfalsifiable - you cannot tell a
 * good ranking from a lucky one. This breaks the number open: which retriever
 * found the paper, at what rank, and what that contributed to the fused score.
 *
 * The case worth looking for is a paper found by exactly one retriever. That is
 * the visible evidence for why fusion exists at all, and it is called out
 * explicitly rather than left for the reader to notice by absence.
 */
export function WhyThisResult({ explanation, fusedScore }: Props) {
  if (explanation.kind === "unknown") {
    return (
      <p className="text-[12.5px] text-muted">
        No explanation available — the backend returned a debug shape this build does not
        recognise.
      </p>
    );
  }

  if (explanation.kind === "single") {
    const isDense = explanation.retriever === "dense";
    return (
      <div className="space-y-2 text-[12.5px]">
        <div className="flex items-baseline justify-between gap-3 font-mono">
          <span>{RETRIEVER_LABEL[explanation.retriever] ?? explanation.retriever}</span>
          <span className="text-muted">
            {isDense ? "cosine" : "BM25"} {explanation.score.toFixed(4)}
          </span>
        </div>
        {explanation.chunkIndex !== null && explanation.chunkIndex > 0 && (
          <p className="text-muted">
            Matched chunk {explanation.chunkIndex + 1} of this abstract. Abstracts over 512
            tokens are split, so a long paper can match on a passage rather than as a whole.
          </p>
        )}
        {explanation.embeddingModel && (
          <p className="font-mono text-[11.5px] text-muted">{explanation.embeddingModel}</p>
        )}
        <p className="text-muted">
          A single-retriever score. It is not comparable to the other mode&rsquo;s numbers —
          BM25 is unbounded and cosine similarity is not a probability, which is exactly why
          fusion works on ranks instead.
        </p>
      </div>
    );
  }

  const missing = missingFrom(explanation, ALL_RETRIEVERS);
  const total = explanation.runs.reduce((sum, run) => sum + run.contribution, 0);

  return (
    <div className="space-y-2.5 text-[12.5px]">
      <table className="w-full font-mono">
        <thead>
          <tr className="text-left text-[11px] uppercase tracking-wide text-muted">
            <th className="pb-1 font-normal">retriever</th>
            <th className="pb-1 text-right font-normal">rank</th>
            <th className="pb-1 text-right font-normal">score</th>
            <th className="pb-1 text-right font-normal">→ contributed</th>
          </tr>
        </thead>
        <tbody>
          {explanation.runs.map((run) => (
            <tr key={run.retriever} className="border-t border-line/70">
              <td className="py-1">{run.retriever}</td>
              <td className="py-1 text-right">{run.rank}</td>
              <td className="py-1 text-right text-muted">{run.score.toFixed(4)}</td>
              <td className="py-1 text-right">{run.contribution.toFixed(5)}</td>
            </tr>
          ))}
          <tr className="border-t border-line">
            <td className="py-1 font-sans font-medium" colSpan={3}>
              fused score
            </td>
            <td className="py-1 text-right font-medium">{total.toFixed(5)}</td>
          </tr>
        </tbody>
      </table>

      <p className="text-muted">
        Each contribution is <code>1 / (k + rank)</code> with k = {explanation.rrfK}. Only the
        rank is used — the raw scores in the middle column are shown for inspection and play no
        part in the arithmetic.
      </p>

      {missing.length > 0 ? (
        <p className="rounded border border-amber-300 bg-amber-50 px-2.5 py-1.5 text-amber-900">
          <span className="font-semibold">Found by {explanation.runs[0]?.retriever} only.</span>{" "}
          {missing.join(" and ")} did not return this paper at all. Results like this one are the
          entire argument for fusing two retrievers — and, when the ranking looks wrong, the
          first place to look.
        </p>
      ) : (
        <p className="text-muted">
          Found by both retrievers. Agreement is the ordinary case; disagreement is the
          interesting one.
        </p>
      )}

      {explanation.degraded.length > 0 && (
        <p className="rounded border border-amber-300 bg-amber-50 px-2.5 py-1.5 text-amber-900">
          {explanation.degraded.join(", ")} failed on this request — this fusion is incomplete.
        </p>
      )}

      <p className="sr-only">Fused score reported by the API: {fusedScore}</p>
    </div>
  );
}
