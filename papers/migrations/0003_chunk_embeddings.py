"""Dense vectors in PostgreSQL.

``VectorExtension`` has to run before the column that depends on it. Django
does not infer that, so this migration is hand-edited: ``makemigrations``
produced the field and the index, and the extension was added above them.

Creating an extension needs superuser rights on the database. That is true of
the CI service container and of a local Homebrew install, and it is worth
knowing before a managed-Postgres deployment where it may not be.
"""

import pgvector.django.indexes
import pgvector.django.vector
from django.db import migrations
from pgvector.django import VectorExtension


class Migration(migrations.Migration):
    dependencies = [
        ("papers", "0002_ingestion_fields"),
    ]

    operations = [
        VectorExtension(),
        migrations.AddField(
            model_name="chunk",
            name="embedding",
            field=pgvector.django.vector.VectorField(
                blank=True,
                dimensions=384,
                help_text=(
                    "The dense vector, L2-normalised. Stored in PostgreSQL beside "
                    "the text it was computed from, so a vector and its source row "
                    "are written in one transaction and cannot drift apart. NULL "
                    "means the chunk has not been embedded yet."
                ),
                null=True,
            ),
        ),
        migrations.AddIndex(
            model_name="chunk",
            index=pgvector.django.indexes.HnswIndex(
                ef_construction=64,
                fields=["embedding"],
                m=16,
                name="chunk_embedding_hnsw",
                opclasses=["vector_cosine_ops"],
            ),
        ),
    ]
