"""Drop the unused second EvaluationRun table.

Two models with this name existed: this one, scaffolded in Phase 1, and the one
in the ``evaluation`` app that the Day 9 harness actually writes to. The API
served this one, so ``/api/evaluation-runs/`` returned an empty list while six
real runs sat in the other table. Dropping it removes the ambiguity rather than
leaving a decoy for the next reader.

The table is empty, so no data is lost.
"""

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('papers', '0003_chunk_embeddings'),
    ]

    operations = [
        migrations.DeleteModel(
            name='EvaluationRun',
        ),
    ]
