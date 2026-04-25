import pytest
from langchain_core.documents import Document
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
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

    chat_history = []
    
    question = "How many contents"

    messages = build_prompt(docs, question, chat_history)

    assert len(messages) == 2
    assert isinstance(messages[0], SystemMessage)
    assert isinstance(messages[1], HumanMessage)

    expected_content = "Content 1\n\nContent 2"
    assert expected_content in messages[1].content
    assert question in messages[1].content

    assert "based only on the provided context" in messages[0].content

def test_build_prompt_logic_chat_history():
    fake_doc = [Document(page_content="Company was founded in 1987")]
    question = "Who is the CEO?"

    chat_history = [
        {"role": "user", "content": "Hello!"},
        {"role": "assistant", "content": "Hi there, how can I help you?"}
    ]

    messages = build_prompt(fake_doc, question, chat_history)

    assert len(messages) == 4

    assert isinstance(messages[0], SystemMessage)

    assert isinstance(messages[1], HumanMessage)
    assert messages[1].content == "Hello!"

    assert isinstance(messages[2], AIMessage)
    assert messages[2].content == "Hi there, how can I help you?"

    assert isinstance(messages[3], HumanMessage)
    assert "Company was founded in 1987" in messages[3].content
    assert "Who is the CEO?" in messages[3].content



def test_generate_answer_delegation(mocker):
    mock_llm = mocker.Mock()
    mock_response = mocker.Mock()
    mock_response.content = "the final answer"
    mock_llm.invoke.return_value = mock_response

    docs = [Document(page_content="context")]
    question = "the question"

    result = generate_answer(question, docs, mock_llm, chat_history=[])

    assert result == "the final answer"

    called_message = mock_llm.invoke.call_args[0][0]
    mock_llm.invoke.assert_called_once()
    assert len(called_message) == 2
    assert "context" in called_message[1].content