import type { SearchMode } from "../api/client";

/**
 * The published Day 9 ablation, shown next to the mode toggle.
 *
 * These are hard-coded on purpose. They are a *finding* from a specific
 * evaluation run against a specific 50-query golden set, not a live metric, and
 * presenting them as though the app computed them just now would be a lie of
 * framing. They change when the evaluation is re-run and republished - not when
 * a user searches.
 */
export interface ModeInfo {
  id: SearchMode;
  label: string;
  blurb: string;
  ndcg: number;
  medianMs: number;
}

export const MODES: ModeInfo[] = [
  {
    id: "bm25",
    label: "BM25",
    blurb: "Keyword. Matches the words you typed.",
    ndcg: 0.473,
    medianMs: 5,
  },
  {
    id: "dense",
    label: "Dense",
    blurb: "Meaning. Embeddings, cosine similarity over pgvector.",
    ndcg: 0.691,
    medianMs: 24,
  },
  {
    id: "hybrid",
    label: "Hybrid",
    blurb: "Both, fused with Reciprocal Rank Fusion at k=60.",
    ndcg: 0.649,
    medianMs: 33,
  },
];

/** The mode the evaluation says is best. Not the most complicated one. */
export const BEST_MODE: SearchMode = "dense";

export const DEFAULT_MODE: SearchMode = BEST_MODE;
export const DEFAULT_TOP_K = 10;

export const EXAMPLE_QUERIES = [
  "how bright was the beam when the collisions were recorded",
  "identifying particles at future colliders",
  "improving a search engine by re-reading its own top results",
  "how do we know dark matter is there if we cannot see it",
];

export const CORPUS = {
  papers: 3377,
  chunks: 3465,
  categories: ["hep-ex", "hep-th", "hep-ph", "cs.IR", "astro-ph.HE"],
};

export const REPO_URL = "https://github.com/Breeganzo/scintilla";
