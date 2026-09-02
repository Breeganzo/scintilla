"""Shared test fixtures."""

from datetime import UTC, datetime

import pytest

from papers.models import Paper


@pytest.fixture
def paper_kwargs() -> dict:
    """Valid field values for building a Paper."""
    return {
        "arxiv_id": "2401.12345v1",
        "title": "Measurement of the Higgs boson coupling to tau leptons",
        "abstract": (
            "We present a measurement of the Higgs boson coupling to tau "
            "leptons using proton-proton collision data. The observed "
            "significance is 5.2 standard deviations."
        ),
        "authors": ["A. Researcher", "B. Collaborator"],
        "categories": ["hep-ex", "hep-ph"],
        "primary_category": "hep-ex",
        "published_at": datetime(2024, 1, 22, 12, 0, tzinfo=UTC),
        "arxiv_updated_at": datetime(2024, 1, 22, 12, 0, tzinfo=UTC),
        "abs_url": "https://arxiv.org/abs/2401.12345v1",
        "pdf_url": "https://arxiv.org/pdf/2401.12345v1",
    }


@pytest.fixture
def paper(db, paper_kwargs) -> Paper:
    return Paper.objects.create(**paper_kwargs)
