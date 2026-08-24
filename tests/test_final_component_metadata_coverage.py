from collections import Counter

from sai.data.institutional_books_cross_source_subdocument_rewrite_aggregate import (
    _metadata_coverage_complete as book_metadata_complete,
)
from sai.data.pleias_cross_source_subdocument_rewrite_aggregate import (
    _metadata_coverage_complete as pleias_metadata_complete,
)


def test_pleias_metadata_coverage_requires_every_single_assignment() -> None:
    totals = Counter(
        {
            "documents": 2,
            "semantic_stratum::reference::documents": 2,
            "quality_floor_milli::4000::documents": 2,
            "curriculum_phase::foundation::documents": 2,
            "difficulty_mean_milli::2500::documents": 2,
            "semantic_domain::science::documents": 1,
            "semantic_domain::history::documents": 1,
        }
    )
    assert pleias_metadata_complete(totals)
    totals["curriculum_phase::foundation::documents"] = 1
    assert not pleias_metadata_complete(totals)


def test_book_metadata_coverage_requires_consensus_for_every_book() -> None:
    totals = Counter(
        {
            "documents": 2,
            "documents_with_consensus_curriculum_metadata": 2,
            "documents_with_quality_floor_metadata": 2,
            "documents_with_complexity_range_metadata": 2,
            "documents_with_translation_type_metadata": 2,
            "semantic_genre::technical_nonfiction::documents": 2,
            "semantic_domain::science::documents": 2,
            "semantic_domain::history::documents": 1,
            "curriculum_band_vote::foundation::documents": 1,
            "curriculum_band_vote::advanced::documents": 1,
            "book_style::instructional::documents": 2,
            "translation_type_vote::none_english::documents": 2,
            "book_quality_floor::overall_quality::4::documents": 2,
            "book_complexity_minimum::conceptual_complexity::2::documents": 2,
            "book_complexity_maximum::conceptual_complexity::3::documents": 2,
        }
    )
    assert book_metadata_complete(totals)
    totals["documents_with_consensus_curriculum_metadata"] = 1
    assert not book_metadata_complete(totals)
