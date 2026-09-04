import { useState } from "react";

import type { SearchMode } from "../api/client";
import { BEST_MODE, MODES } from "../lib/modes";

interface Props {
  mode: SearchMode;
  onChange: (mode: SearchMode) => void;
}

/**
 * The retrieval strategy, exposed rather than hidden.
 *
 * Most search interfaces would never surface this. It is surfaced here because
 * `mode` is the ablation mechanism - the evaluation harness calls the same
 * endpoint once per mode - so the thing that was measured is the thing you are
 * using. Hiding it would make the published table a claim about some other
 * system.
 *
 * The nDCG figure sits under each option for one reason: it stops the interface
 * from implying that the most elaborate strategy is the best one. It is not.
 */
export function ModeToggle({ mode, onChange }: Props) {
  const [open, setOpen] = useState(false);

  return (
    <div>
      <div role="radiogroup" aria-label="Retrieval mode" className="flex flex-wrap gap-2">
        {MODES.map((info) => {
          const selected = info.id === mode;
          return (
            <button
              key={info.id}
              type="button"
              role="radio"
              aria-checked={selected}
              onClick={() => onChange(info.id)}
              className={[
                "flex-1 basis-52 rounded-lg border px-3.5 py-2.5 text-left transition-colors",
                selected
                  ? "border-accent bg-accent/5 ring-1 ring-accent/25"
                  : "border-line hover:border-muted/40",
              ].join(" ")}
            >
              <div className="flex items-baseline justify-between gap-2">
                <span className="text-[15px] font-medium">{info.label}</span>
                {info.id === BEST_MODE && (
                  <span className="rounded-full bg-accent/10 px-2 py-0.5 text-[11px] font-medium text-accent">
                    measured best
                  </span>
                )}
              </div>
              <p className="mt-0.5 text-[12.5px] leading-snug text-muted">{info.blurb}</p>
              <p className="mt-1.5 font-mono text-[11.5px] text-muted">
                nDCG@10 {info.ndcg.toFixed(3)} · {info.medianMs} ms retrieval
              </p>
            </button>
          );
        })}
      </div>

      {/* Two different timings appear on this page and they do not agree. Saying
          so is cheaper than letting someone notice and conclude one of them is
          made up. */}
      <p className="mt-2 text-[12px] leading-relaxed text-muted">
        Figures above come from the published 50-query evaluation, which times retrieval
        in-process. The timing under your results is measured end-to-end through HTTP, so it is
        larger — and the first query after a restart is larger again, because the embedding model
        loads on demand.
      </p>

      <div className="mt-2.5 rounded-lg border border-line bg-surface px-3.5 py-2.5">
        <p className="text-[13px] leading-relaxed text-muted">
          <span className="font-medium text-ink">Dense wins, and hybrid does not.</span>{" "}
          Hybrid − dense is −0.041 nDCG@10, 95% CI [−0.103, +0.020], p = 0.194 over 40 queries.
          That is not &ldquo;hybrid is worse&rdquo; — it is{" "}
          <em>we cannot tell them apart, and the simpler system is faster</em>.{" "}
          <button
            type="button"
            onClick={() => setOpen((value) => !value)}
            className="font-medium text-accent underline-offset-2 hover:underline"
          >
            {open ? "Less" : "Why the architecture was not quietly changed to match"}
          </button>
        </p>

        {open && (
          <div className="mt-2.5 space-y-2 border-t border-line pt-2.5 text-[13px] leading-relaxed text-muted">
            <p>
              Reciprocal Rank Fusion only sees <em>ranks</em>, never scores, so it weights both
              retrievers equally and cannot know that BM25 is nearly useless on paraphrased
              queries. On that class BM25 scores 0.206 recall@10 and dense 0.615 — fusing them
              lands at 0.397, in between, which is worse than not fusing at all.
            </p>
            <p>
              Sweeping the RRF constant <code className="font-mono">k</code> shows a monotonic
              decline: k=5 gives nDCG 0.676 against the shipped default of 60 at 0.649. The
              default is close to the worst setting for this corpus.
            </p>
            <p>
              <span className="font-medium text-ink">k was deliberately left at 60.</span> Tuning
              a hyper-parameter on the same 40 queries the result is then published against is
              overfitting. The sweep ships as a diagnosis, not as a result.
            </p>
          </div>
        )}
      </div>
    </div>
  );
}
