import pytest
from langchain_core.documents import Document
from langchain_core.messages import SystemMessage, HumanMessage
from src.generation.llm import get_llm, build_prompt, generate_answer

def test_get_llm_missing_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(ValueError, match = "ANTHROPIC_API_KEY must be set"):
        get_llm()

def test_get_llm_config(mocker, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setenv("ANTHROPIC_MODEL", "claude-test")
    monkeypatch.setenv("MAX_TOKENS", "500")

    mock_chat = mocker.patch("src.generation.llm.ChatAnthropic")

    get_llm()

    _, kwargs = mock_chat.call_args
    assert kwargs["model"] == "claude-test"
    assert kwargs["max_tokens"] == 500
    assert kwargs["anthropic_api_key"] == "test-key"

def test_build_prompt_logic():
    docs = [
        Document(page_content="Content 1"),
        Document(page_content="Content 2")
    ]

    question = "How many contents"

    messages = build_prompt(docs, question)

    assert len(messages) == 2
    assert isinstance(messages[0], SystemMessage)
    assert isinstance(messages[1], HumanMessage)

    expected_content = "Content 1\n\nContent 2"
    assert expected_content in messages[1].content
    assert question in messages[1].content

    assert "based only on the provided context" in messages[0].content

def test_generate_answer_delegation(mocker):
    mock_llm = mocker.Mock()
    mock_response = mocker.Mock()
    mock_response.content = "the final answer"
    mock_llm.invoke.return_value = mock_response

    docs = [Document(page_content="context")]
    question = "the question"

    result = generate_answer(question, docs, mock_llm)

    assert result == "the final answer"

    called_message = mock_llm.invoke.call_args[0][0]
    mock_llm.invoke.assert_called_once()
    assert len(called_message) == 2
    assert "context" in called_message[1].content
    
