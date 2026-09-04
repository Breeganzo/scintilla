"""Check the golden set against the live corpus.

Separate from the test suite because it needs the real collection, not a test
database. A label pointing at a paper that is not in the corpus scores zero for
every retriever forever, which in an ablation table looks exactly like a query
that is merely hard - so this failure has to be caught deliberately or not at
all.
"""

from django.core.management.base import BaseCommand, CommandError

from evaluation.golden import load_golden_set
from papers.models import Paper


class Command(BaseCommand):
    help = "Verify every labelled paper in the golden set exists in the corpus."

    def add_arguments(self, parser):
        # Not --version: Django's BaseCommand already owns that flag.
        parser.add_argument("--golden-version", type=int, default=1)

    def handle(self, *args, **options):
        golden = load_golden_set(version=options["golden_version"])
        labelled = golden.relevant_ids

        titles = dict(Paper.objects.filter(arxiv_id__in=labelled).values_list("arxiv_id", "title"))
        missing = sorted(labelled - set(titles))

        self.stdout.write(
            f"golden set v{golden.version}: {len(golden)} queries, "
            f"{len(labelled)} distinct labelled papers"
        )

        # A query whose text contains its answer's title is a gift to lexical
        # retrieval and would inflate BM25 for reasons unrelated to quality.
        quoting = [
            (query.id, paper_id)
            for query in golden
            for paper_id in query.relevant_ids
            if (title := titles.get(paper_id))
            and len(title) > 20
            and title.lower() in query.query.lower()
        ]

        for query in golden:
            absent = [pid for pid in query.relevant_ids if pid in missing]
            if absent:
                self.stdout.write(self.style.ERROR(f"  {query.id}: {' '.join(absent)}"))

        if quoting:
            for query_id, paper_id in quoting:
                self.stdout.write(self.style.ERROR(f"  {query_id} quotes title of {paper_id}"))

        if missing or quoting:
            raise CommandError(f"{len(missing)} missing, {len(quoting)} quoting a title")

        self.stdout.write(self.style.SUCCESS("all labels resolve to papers in the corpus"))
