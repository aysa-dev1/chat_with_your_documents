from langchain_core.documents import Document

from src.retrieval.vector_store import ChromaVectorStore


def test_add_documents_delegation(mocker):

    mock_chroma_cls = mocker.patch("src.retrieval.vector_store.Chroma")
    mock_embedder = mocker.Mock()

    vector_store = ChromaVectorStore(embedder=mock_embedder, collection_name="test_col")

    docs = [Document(page_content="test content", metadata={"source": "test.pdf"})]
    vector_store.add_documents(docs)

    mock_chroma_cls.return_value.add_documents.assert_called_once_with(docs)


def test_similarity_search_returns_results(mocker):

    mock_chroma_cls = mocker.patch("src.retrieval.vector_store.Chroma")
    vector_store = ChromaVectorStore(mocker.Mock(), "test_col")

    expected_docs = [Document(page_content="found text")]
    mock_chroma_cls.return_value.similarity_search.return_value = expected_docs

    results = vector_store.similarity_search("query text", k=5)

    mock_chroma_cls.return_value.similarity_search.assert_called_once_with(
        "query text", k=5
    )
    assert results == expected_docs


def test_persist_directory_env_var(mocker, monkeypatch):
    mock_chroma_cls = mocker.patch("src.retrieval.vector_store.Chroma")
    test_path = "/tmp/test-chroma_path"
    monkeypatch.setenv("CHROMA_PERSIST_DIR", test_path)

    ChromaVectorStore(mocker.Mock(), "test_col")

    _, kwargs = mock_chroma_cls.call_args
    assert kwargs["persist_directory"] == test_path


def test_persist_directory_default(mocker, monkeypatch):
    mock_chroma_cls = mocker.patch("src.retrieval.vector_store.Chroma")
    monkeypatch.delenv("CHROMA_PERSIST_DIR", raising=False)

    ChromaVectorStore(mocker.Mock(), "test_col")

    _, kwargs = mock_chroma_cls.call_args
    assert kwargs["persist_directory"] == "./data/chroma"
