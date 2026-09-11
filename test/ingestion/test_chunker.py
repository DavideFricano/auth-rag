from datetime import date

import pytest

from auth_rag.ingestion.chunker import (
    FixedSizeChunker,
    HierarchicalChunker,
    RecursiveCharacterChunker,
    SemanticChunker,
    SentenceChunker,
)
from auth_rag.types import Document, Origin, Source


def _doc(text: str) -> Document:
    return Document(
        text=text, source=Source(id="doc1", name="doc.pdf", origin=Origin.LOCAL, time=date(2024, 1, 1))
    )


# --- HierarchicalChunker ---

def test_hierarchical_empty_document_returns_no_chunks():
    assert HierarchicalChunker().chunk(_doc("")) == []

def test_hierarchical_chunk_ids_unique():
    chunks = HierarchicalChunker().chunk(_doc("# H1\n\nParagrafo uno.\n\n## H2\n\nParagrafo due."))
    ids = [c.id for c in chunks]
    assert len(ids) == len(set(ids))

def test_hierarchical_chunk_id_contains_source_id():
    chunks = HierarchicalChunker().chunk(_doc("# Titolo\n\nTesto."))
    assert all("doc1" in c.id for c in chunks)

def test_hierarchical_metadata_source_matches_document():
    chunks = HierarchicalChunker().chunk(_doc("# Titolo\n\nTesto."))
    assert all(c.metadata.source.name == "doc.pdf" for c in chunks)

def test_hierarchical_section_path_contains_heading():
    chunks = HierarchicalChunker().chunk(_doc("# Introduzione\n\nTesto."))
    assert any("Introduzione" in c.metadata.title for c in chunks)

def test_hierarchical_nested_section_path():
    chunks = HierarchicalChunker().chunk(_doc("# Cap1\n\n## Sez1\n\nTesto."))
    assert any("Cap1" in c.metadata.title and "Sez1" in c.metadata.title for c in chunks)

def test_hierarchical_body_inherits_section():
    chunks = HierarchicalChunker().chunk(_doc("# Titolo\n\nParagrafo sotto il titolo."))
    body_chunks = [c for c in chunks if "Paragrafo" in c.text]
    assert all(c.metadata.title != "" for c in body_chunks)

def test_hierarchical_heading_stripped_from_text():
    chunks = HierarchicalChunker().chunk(_doc("# Titolo\n\nContenuto del paragrafo."))
    texts = [c.text for c in chunks]
    assert all("# Titolo" not in t for t in texts)
    assert any("Contenuto del paragrafo" in t for t in texts)


# --- FixedSizeChunker ---

def test_fixed_size_invalid_overlap_raises():
    with pytest.raises(ValueError):
        FixedSizeChunker(chunk_size=10, overlap=10)

def test_fixed_size_empty_returns_no_chunks():
    assert FixedSizeChunker().chunk(_doc("   ")) == []

def test_fixed_size_chunk_size_respected():
    chunks = FixedSizeChunker(chunk_size=20, overlap=0).chunk(_doc("a" * 100))
    assert all(len(c.text) <= 20 for c in chunks)

def test_fixed_size_first_chunk_text():
    chunks = FixedSizeChunker(chunk_size=10, overlap=0).chunk(_doc("abcdefghijklmnopqrst"))
    assert chunks[0].text == "abcdefghij"
    assert chunks[1].text == "klmnopqrst"

def test_fixed_size_overlap_shares_text():
    chunks = FixedSizeChunker(chunk_size=20, overlap=5).chunk(_doc("abcdefghijklmnopqrstuvwxyz" * 4))
    assert chunks[0].text[15:20] == chunks[1].text[:5]

def test_fixed_size_ids_unique_for_distinct_text():
    chunks = FixedSizeChunker(chunk_size=10, overlap=2).chunk(_doc("abcdefghijklmnopqrstuvwxyz" * 4))
    ids = [c.id for c in chunks]
    assert len(ids) == len(set(ids))

def test_fixed_size_ids_collide_for_identical_text():
    """Content-addressed ids: identical chunk text within a document dedups to the same id."""
    chunks = FixedSizeChunker(chunk_size=10, overlap=2).chunk(_doc("a" * 100))
    ids = {c.id for c in chunks}
    assert len(ids) == len({c.text for c in chunks})


# --- SentenceChunker ---

def test_sentence_invalid_overlap_raises():
    with pytest.raises(ValueError):
        SentenceChunker(sentences_per_chunk=3, overlap=3)

def test_sentence_preserves_sentence_text():
    text = "Prima frase. Seconda frase. Terza frase. Quarta frase."
    chunks = SentenceChunker(sentences_per_chunk=2, overlap=0).chunk(_doc(text))
    assert "Prima frase" in chunks[0].text
    assert "Seconda frase" in chunks[0].text
    assert "Terza frase" in chunks[1].text
    assert "Quarta frase" in chunks[1].text

def test_sentence_overlap_shares_sentence():
    text = "Prima frase. Seconda frase. Terza frase. Quarta frase."
    chunks = SentenceChunker(sentences_per_chunk=2, overlap=1).chunk(_doc(text))
    # "Seconda frase" deve comparire sia nel chunk 0 che nel chunk 1
    assert "Seconda frase" in chunks[0].text
    assert "Seconda frase" in chunks[1].text

def test_sentence_overlap_produces_more_chunks():
    text = " ".join([f"Frase numero {i}." for i in range(9)])
    no_overlap = SentenceChunker(sentences_per_chunk=3, overlap=0).chunk(_doc(text))
    with_overlap = SentenceChunker(sentences_per_chunk=3, overlap=1).chunk(_doc(text))
    assert len(with_overlap) > len(no_overlap)

def test_sentence_ids_unique():
    text = " ".join([f"Frase numero {i}." for i in range(10)])
    chunks = SentenceChunker(sentences_per_chunk=2, overlap=0).chunk(_doc(text))
    ids = [c.id for c in chunks]
    assert len(ids) == len(set(ids))


# --- RecursiveCharacterChunker ---

def test_recursive_invalid_overlap_raises():
    with pytest.raises(ValueError):
        RecursiveCharacterChunker(chunk_size=10, overlap=10)

def test_recursive_empty_returns_no_chunks():
    assert RecursiveCharacterChunker().chunk(_doc("   ")) == []

def test_recursive_respects_chunk_size():
    text = "Paragrafo uno.\n\nParagrafo due.\n\nParagrafo tre.\n\nParagrafo quattro."
    chunks = RecursiveCharacterChunker(chunk_size=20, overlap=0).chunk(_doc(text))
    assert all(len(c.text) <= 20 for c in chunks)

def test_recursive_prefers_paragraph_boundary():
    text = "Primo paragrafo con testo.\n\nSecondo paragrafo con testo."
    chunks = RecursiveCharacterChunker(chunk_size=512, overlap=0).chunk(_doc(text))
    assert len(chunks) == 1

def test_recursive_overlap_shares_text():
    long_word = "a" * 30
    text = f"{long_word} {long_word} {long_word} {long_word}"
    chunks = RecursiveCharacterChunker(chunk_size=40, overlap=10).chunk(_doc(text))
    assert len(chunks) > 1
    assert chunks[0].text[-10:] in chunks[1].text

def test_recursive_ids_unique():
    text = "\n\n".join([f"Paragrafo {i} con contenuto." for i in range(10)])
    chunks = RecursiveCharacterChunker(chunk_size=30, overlap=5).chunk(_doc(text))
    ids = [c.id for c in chunks]
    assert len(ids) == len(set(ids))

def test_recursive_custom_separators():
    text = "parte1|parte2|parte3"
    chunks = RecursiveCharacterChunker(chunk_size=10, overlap=0, separators=["|", ""]).chunk(_doc(text))
    assert any("parte1" in c.text for c in chunks)
    assert any("parte2" in c.text for c in chunks)


# --- SemanticChunker ---

def test_semantic_empty_returns_no_chunks():
    assert SemanticChunker().chunk(_doc("")) == []

def test_semantic_single_sentence_returns_one_chunk():
    chunks = SemanticChunker().chunk(_doc("Una sola frase."))
    assert len(chunks) == 1
    assert "Una sola frase" in chunks[0].text

def test_semantic_ids_unique():
    text = " ".join([f"Frase numero {i} con contenuto diverso." for i in range(10)])
    chunks = SemanticChunker().chunk(_doc(text))
    ids = [c.id for c in chunks]
    assert len(ids) == len(set(ids))

def test_semantic_metadata_source_matches():
    chunks = SemanticChunker().chunk(_doc("Prima frase. Seconda frase."))
    assert all(c.metadata.source.name == "doc.pdf" for c in chunks)

def test_semantic_max_chunk_size_respected():
    text = " ".join([f"Frase {i}." for i in range(50)])
    chunks = SemanticChunker(max_chunk_size=100).chunk(_doc(text))
    assert all(len(c.text) <= 100 for c in chunks)

def test_semantic_low_threshold_produces_more_chunks():
    text = "Il gatto dorme sul divano. La macchina è parcheggiata in garage. Il sole tramonta a ovest."
    chunks_low = SemanticChunker(breakpoint_threshold=0.99).chunk(_doc(text))
    chunks_high = SemanticChunker(breakpoint_threshold=0.0).chunk(_doc(text))
    assert len(chunks_low) >= len(chunks_high)
