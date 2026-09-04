"""Measure the answering layer over the golden set.

Separate from ``run_eval`` because the two have different costs and different
failure modes. Retrieval evaluation is free, deterministic and runs in seconds,
so CI can run it on every push. Answering evaluation calls a hosted model fifty
times, costs tokens, depends on a third party being up, and is not perfectly
reproducible even at temperature zero. Folding them into one command would drag
those properties onto the cheap measurement and make the regression gate
flaky - and a gate that fails for reasons unrelated to the code is a gate that
gets switched off.
"""

from __future__ import annotations

import json
from pathlib import Path

from django.conf import settings
from django.core.management.base import BaseCommand, CommandError

from evaluation.answer_harness import run_answer_eval
from evaluation.answer_report import to_dict, to_markdown
from evaluation.golden import load_golden_set
from evaluation.management.commands.run_eval import current_git_sha
from search.answering import DEFAULT_CONTEXT_SIZE
from search.llm import get_provider
from search.retrievers import DEFAULT_TOP_K

DEFAULT_OUTPUT_DIR = Path(settings.BASE_DIR) / "evaluation" / "results"


class Command(BaseCommand):
    help = "Run the golden set through retrieval and answering, and report abstention."

    def add_arguments(self, parser):
        parser.add_argument("--mode", default="dense")
        parser.add_argument("--golden-version", type=int, default=1)
        parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
        parser.add_argument("--context-size", type=int, default=DEFAULT_CONTEXT_SIZE)
        parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
        parser.add_argument(
            "--pause",
            type=float,
            default=0.5,
            help="Seconds between model calls, to stay under a free-tier rate limit.",
        )
        parser.add_argument("--no-save", action="store_true")
        parser.add_argument("--no-warmup", action="store_true")

    def handle(self, *args, **options):
        provider = get_provider()
        if not provider.available:
            # Refusing rather than emitting a report of zeros. A file named
            # answers.json containing a 0% abstention rate, produced because no
            # model was configured, is worse than no file: it looks like a
            # measurement and it is not one.
            raise CommandError(
                "No LLM provider is configured, so there is nothing to measure. "
                "Set GROQ_API_KEY in .env and try again."
            )

        golden = load_golden_set(version=options["golden_version"])
        self.stdout.write(
            f"Answering {len(golden)} golden queries with {provider.name} "
            f"over retriever '{options['mode']}'..."
        )

        report = run_answer_eval(
            mode=options["mode"],
            golden=golden,
            top_k=options["top_k"],
            context_size=options["context_size"],
            provider=provider,
            warm_up=not options["no_warmup"],
            pause_seconds=options["pause"],
        )

        if report.degraded:
            degraded = [o.query_id for o in report.outcomes if o.result.degraded]
            raise CommandError(
                f"{len(degraded)} queries degraded ({', '.join(degraded[:5])}...). "
                "The report would understate the abstention rate, so it was not written."
            )

        markdown = to_markdown(report)
        self.stdout.write("\n" + markdown)

        if options["no_save"]:
            return

        output_dir: Path = options["output_dir"]
        output_dir.mkdir(parents=True, exist_ok=True)
        payload = to_dict(report)
        payload["git_sha"] = current_git_sha()

        (output_dir / "answers.json").write_text(json.dumps(payload, indent=2) + "\n")
        (output_dir / "answers.md").write_text(markdown)
        self.stdout.write(self.style.SUCCESS(f"Wrote answers.json and answers.md to {output_dir}"))
