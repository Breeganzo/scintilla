import { useEffect, useRef, useState } from "react";

import { EXAMPLE_QUERIES } from "../lib/modes";

interface Props {
  initialQuery: string;
  busy: boolean;
  onSearch: (query: string) => void;
}

export function SearchBar({ initialQuery, busy, onSearch }: Props) {
  const [value, setValue] = useState(initialQuery);
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    inputRef.current?.focus();
  }, []);

  const submit = (query: string) => {
    const trimmed = query.trim();
    // The backend rejects a blank query with a 400. Not sending it at all is
    // not "client-side validation instead of server-side" - the server check
    // stays. This just avoids spending a rate-limit token to be told no.
    if (!trimmed) return;
    onSearch(trimmed);
  };

  return (
    <div>
      <form
        onSubmit={(event) => {
          event.preventDefault();
          submit(value);
        }}
        className="flex gap-2"
      >
        <input
          ref={inputRef}
          type="search"
          value={value}
          onChange={(event) => setValue(event.target.value)}
          placeholder="Ask in your own words — the corpus is particle physics, cosmology and information retrieval"
          aria-label="Search query"
          maxLength={512}
          className="min-w-0 flex-1 rounded-lg border border-line bg-white px-4 py-3 text-[15px] outline-none placeholder:text-muted/70 focus:border-accent focus:ring-2 focus:ring-accent/20"
        />
        <button
          type="submit"
          disabled={busy || !value.trim()}
          className="shrink-0 rounded-lg bg-ink px-5 py-3 text-[15px] font-medium text-white transition-opacity disabled:opacity-40"
        >
          {busy ? "Searching…" : "Search"}
        </button>
      </form>

      <div className="mt-2.5 flex flex-wrap items-center gap-x-2 gap-y-1.5 text-[13px]">
        <span className="text-muted">Try:</span>
        {EXAMPLE_QUERIES.map((example) => (
          <button
            key={example}
            type="button"
            onClick={() => {
              setValue(example);
              submit(example);
            }}
            className="rounded-full border border-line px-2.5 py-1 text-muted transition-colors hover:border-accent hover:text-accent"
          >
            {example}
          </button>
        ))}
      </div>
    </div>
  );
}
