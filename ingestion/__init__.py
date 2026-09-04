"""Harvesting arXiv into the corpus.

Deliberately kept out of the ``papers`` Django app. ``papers`` owns what the
data *is* - models, serializers, endpoints. ``ingestion`` owns how it *gets
there*. Keeping them apart means the API can be reasoned about without reading
pipeline code, and the pipeline can be unit tested without a web request.
"""
