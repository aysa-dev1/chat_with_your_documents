from langchain_anthropic import ChatAnthropic
from langchain_core.documents import Document
from langchain_core.messages import SystemMessage, HumanMessage, BaseMessage, AIMessage
import os

def get_llm() -> ChatAnthropic:

    model_name = os.getenv("ANTHROPIC_MODEL", "claude-opus-4-7")
    max_tokens = int(os.getenv("MAX_TOKENS", "1024"))
    anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")

    if not anthropic_api_key:
        raise ValueError("ANTHROPIC_API_KEY must be set in the environment")

    return ChatAnthropic(
        model=model_name,
        max_tokens=max_tokens,
        anthropic_api_key=anthropic_api_key
    )


def build_prompt(context_docs: list[Document], question: str, chat_history: list[dict[str, str]]) -> list[BaseMessage]:
    
    messages = []

    system_content = (
        "You are a helpful assistant having a conversation about a document. Answer questions naturally and directly based only on the provided context. "
        "If the information is not in the context, say so briefly."
    )
    messages.append(SystemMessage(content=system_content))

    if chat_history is not None:
        for entry in chat_history:
            if entry["role"] == "user":
                messages.append(HumanMessage(content=entry["content"]))
            elif entry["role"] == "assistant":
                messages.append(AIMessage(content=entry["content"]))

    context_text = "\n\n".join([doc.page_content for doc in context_docs])
    human_content = (
        f"Context:\n{context_text}\n\n"
        f"Question: {question}"
    )

    messages.append(HumanMessage(content=human_content))

    return messages


def generate_answer(question: str, context_docs: list[Document], llm: ChatAnthropic, chat_history: list[dict[str, str]]=None) -> str:
    
    messages = build_prompt(context_docs, question, chat_history)

    response = llm.invoke(messages)

    return str(response.content)
