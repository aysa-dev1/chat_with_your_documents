from src.retrieval.embedder import get_embedder


def test_get_embedder_config(mocker):
    mock_hf = mocker.patch("src.retrieval.embedder.HuggingFaceEmbeddings")

    embedder = get_embedder()

    assert embedder == mock_hf.return_value

    args, kwargs = mock_hf.call_args

    assert kwargs["model_name"] == "sentence-transformers/all-MiniLM-L6-v2"
    assert kwargs["model_kwargs"] == {"device": "cpu"}
    assert kwargs["encode_kwargs"] == {"normalize_embeddings": True}
