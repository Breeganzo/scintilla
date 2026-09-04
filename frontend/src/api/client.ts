/**
 * The only place in the app that talks to Django.
 *
 * Two rules hold here and nowhere else:
 *
 *   1. Types come from `schema.d.ts`, which is generated from the backend's own
 *      OpenAPI document. Nothing in this file redeclares the response shape. If
 *      a serializer changes and the frontend is not updated, `tsc --noEmit` in
 *      CI fails - which turns a class of runtime bug into a build error.
 *
 *   2. Failure is a *value*, not an exception. Every call returns a
 *      discriminated union. A caller cannot render the success branch without
 *      first proving it is on the success branch, so "forgot to handle the
 *      error" is not expressible. Throwing would let a 429 from the search
 *      throttle surface as a blank page and a console message nobody reads.
 */

import type { components } from "./schema";

export type SearchResult = components["schemas"]["SearchResult"];
export type SearchResponse = components["schemas"]["SearchResponse"];
export type SearchMode = components["schemas"]["SearchRequestModeEnum"];

/** Distinguishable failures. The UI says something different for each. */
export type ApiFailureKind =
  | "network" // never reached the server
  | "throttled" // 429 - our own rate limit, and a normal thing to hit
  | "invalid" // 400 - the request was rejected, message comes from DRF
  | "server"; // 5xx or anything else

export type ApiResult<T> =
  | { ok: true; data: T }
  | { ok: false; kind: ApiFailureKind; message: string; status: number | null };

/**
 * Requests are aborted rather than left running. Without this, typing quickly
 * leaves several dense searches in flight and whichever returns last wins -
 * so the results on screen can belong to a query the user has already replaced.
 * That bug looks like "the ranking is wrong", which is the most expensive kind
 * of bug to misdiagnose in a project whose whole claim is about ranking.
 */
export interface SearchParams {
  query: string;
  mode: SearchMode;
  topK: number;
  signal?: AbortSignal;
}

const SEARCH_URL = "/api/search/";

function messageFromBody(body: unknown, fallback: string): string {
  if (typeof body !== "object" || body === null) return fallback;
  const record = body as Record<string, unknown>;

  // DRF field errors: {"query": ["Query must not be blank."]}
  for (const value of Object.values(record)) {
    if (Array.isArray(value) && typeof value[0] === "string") return value[0];
    if (typeof value === "string") return value;
  }
  return fallback;
}

export async function search(params: SearchParams): Promise<ApiResult<SearchResponse>> {
  let response: Response;
  try {
    response = await fetch(SEARCH_URL, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        query: params.query,
        mode: params.mode,
        top_k: params.topK,
      }),
      ...(params.signal ? { signal: params.signal } : {}),
    });
  } catch (error) {
    // An aborted request is not a failure the user should ever see; the caller
    // re-throws it so the stale render is simply dropped.
    if (error instanceof DOMException && error.name === "AbortError") throw error;
    return {
      ok: false,
      kind: "network",
      status: null,
      message: "Could not reach the API. Is the Django server running on :8001?",
    };
  }

  let body: unknown = null;
  try {
    body = await response.json();
  } catch {
    body = null;
  }

  if (response.ok) {
    return { ok: true, data: body as SearchResponse };
  }

  if (response.status === 429) {
    return {
      ok: false,
      kind: "throttled",
      status: 429,
      message: messageFromBody(body, "Too many searches. Give it a few seconds."),
    };
  }

  if (response.status === 400) {
    return {
      ok: false,
      kind: "invalid",
      status: 400,
      message: messageFromBody(body, "That query was rejected."),
    };
  }

  return {
    ok: false,
    kind: "server",
    status: response.status,
    message: messageFromBody(body, `The API returned ${response.status}.`),
  };
}
