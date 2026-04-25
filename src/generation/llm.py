from langchain_anthropic import ChatAnthropic
from langchain_core.documents import Document
from langchain_core.messages import SystemMessage, HumanMessage, BaseMessage
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


def build_prompt(context_docs: list[Document], question: str) -> list[BaseMessage]:
    
    context_text = "\n\n".join([doc.page_content for doc in context_docs])

    system_content = (
        "You are a helpful assistant having a conversation about a document. Answer questions naturally and directly based only on the provided context. "
        "If the information is not in the context, say so briefly."
    )

    human_content = (
        f"Context:\n{context_text}\n\n"
        f"Question: {question}"
    )

    return [
        SystemMessage(content=system_content),
        HumanMessage(content=human_content)
    ]


def generate_answer(question: str, context_docs: list[Document], llm: ChatAnthropic) -> str:
    
    messages = build_prompt(context_docs, question)

    response = llm.invoke(messages)

    return str(response.content)
