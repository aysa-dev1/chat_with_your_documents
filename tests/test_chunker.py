from langchain_core.documents import Document

from src.ingestion.chunker import chunk_documents


def test_chunk_documents_valid_documents(monkeypatch):
    monkeypatch.setenv("CHUNK_SIZE", "50")
    monkeypatch.setenv("CHUNK_OVERLAP", "10")

    long_text = "ABCD" * 100
    doc = Document(page_content=long_text, metadata={"source": "test.pdf", "page": 1})

    chunks = chunk_documents([doc])

    assert len(chunks) > 1
    for chunk in chunks:
        assert len(chunk.page_content) <= 50


def test_chunk_documents_metadata_preservation(monkeypatch):
    monkeypatch.setenv("CHUNK_SIZE", "20")
    monkeypatch.setenv("CHUNK_OVERLAP", "0")

    original_metadata = {"source": "manual.pdf", "page": 42, "author": "aysa-dev"}

    doc = Document(page_content="AHBKLSOMD" * 5, metadata=original_metadata)

    chunks = chunk_documents([doc])

    assert len(chunks) >= 2
    for chunk in chunks:
        assert chunk.metadata["source"] == "manual.pdf"
        assert chunk.metadata["page"] == 42
        assert chunk.metadata["author"] == "aysa-dev"


def test_chunk_documents_empty_list():

    result = chunk_documents([])

    assert result == []
