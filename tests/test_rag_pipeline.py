from src.services.rag_pipeline import run_rag_pipeline

def test_run_rag_pipeline_valid_path(mocker):
    mock_vector_store = mocker.Mock()
    mock_llm = mocker.Mock()

    mock_results = [mocker.Mock(), mocker.Mock()]
    mock_vector_store.similarity_search.return_value = mock_results

    mock_gen = mocker.patch(
        "src.services.rag_pipeline.generate_answer",
        return_value="mocked answer"
    )

    question = "How do I test RAG?"
    chat_history = []
    answer, sources = run_rag_pipeline(question, mock_vector_store, mock_llm, chat_history)

    mock_vector_store.similarity_search.assert_called_once_with(question, 4)

    mock_gen.assert_called_once_with(question, mock_results, mock_llm, chat_history)

    assert answer == "mocked answer"
    assert sources == mock_results


def test_run_rag_pipeline_custom_k(mocker, monkeypatch):
    monkeypatch.setenv("TOP_K_RESULTS", "7")
    mock_vector_store = mocker.Mock()
    mock_llm = mocker.Mock()

    mocker.patch("src.services.rag_pipeline.generate_answer")

    run_rag_pipeline("Question", mock_vector_store, mock_llm)

    assert mock_vector_store.similarity_search.call_args[0][1] == 7 
