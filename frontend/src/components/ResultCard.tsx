import { useState } from "react";

import type { SearchResult } from "../api/client";
import { explain } from "../api/explain";
import { WhyThisResult } from "./WhyThisResult";

interface Props {
  result: SearchResult;
}

function formatAuthors(authors: string[]): string {
  // hep-ex collaboration papers routinely carry 2,000+ authors. Rendering them
  // all is not thoroughness, it is a wall that hides the title above it.
  if (authors.length === 0) return "Unknown authors";
  const shown = authors.slice(0, 3).join(", ");
  const remaining = authors.length - 3;
  return remaining > 0 ? `${shown} +${remaining}` : shown;
}

function formatDate(iso: string): string {
  const date = new Date(iso);
  return Number.isNaN(date.getTime()) ? iso : date.toISOString().slice(0, 10);
}

export function ResultCard({ result }: Props) {
  const [open, setOpen] = useState(false);
  const explanation = explain(result.debug);

  return (
    <li className="rounded-xl border border-line bg-white p-4 transition-colors hover:border-muted/35">
      <div className="flex gap-3.5">
        <span
          aria-hidden
          className="mt-0.5 w-6 shrink-0 font-mono text-[13px] tabular-nums text-muted"
        >
          {result.rank}
        </span>

        <div className="min-w-0 flex-1">
          <div className="flex items-baseline justify-between gap-3">
            <h3 className="text-[15.5px] font-medium leading-snug">
              <a
                href={result.abs_url}
                target="_blank"
                rel="noopener noreferrer"
                className="underline-offset-2 hover:text-accent hover:underline"
              >
                {result.title}
              </a>
            </h3>
            <span className="shrink-0 font-mono text-[12px] tabular-nums text-muted">
              {result.score.toFixed(4)}
            </span>
          </div>

          <p className="mt-1 text-[12.5px] text-muted">
            {formatAuthors(result.authors)}
            <span className="mx-1.5" aria-hidden>
              ·
            </span>
            <span className="font-mono">{result.primary_category}</span>
            <span className="mx-1.5" aria-hidden>
              ·
            </span>
            {formatDate(result.published_at)}
            <span className="mx-1.5" aria-hidden>
              ·
            </span>
            <span className="font-mono">{result.arxiv_id}</span>
          </p>

          <p className="mt-2 line-clamp-3 text-[13.5px] leading-relaxed text-ink/80">
            {result.abstract}
          </p>

          <div className="mt-2.5 flex items-center gap-3">
            <button
              type="button"
              onClick={() => setOpen((value) => !value)}
              aria-expanded={open}
              className="text-[12.5px] font-medium text-accent underline-offset-2 hover:underline"
            >
              {open ? "▾ Hide explanation" : "▸ Why this result"}
            </button>
            <a
              href={result.abs_url}
              target="_blank"
              rel="noopener noreferrer"
              className="text-[12.5px] text-muted underline-offset-2 hover:text-accent hover:underline"
            >
              arXiv ↗
            </a>
          </div>

          {open && (
            <div className="mt-2.5 rounded-lg border border-line bg-surface p-3">
              <WhyThisResult explanation={explanation} fusedScore={result.score} />
            </div>
          )}
        </div>
      </div>
    </li>
  );
}
