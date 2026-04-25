import os
from typing import Optional

from langchain_anthropic import ChatAnthropic

from src.generation.llm import generate_answer
from src.retrieval.vector_store import VectorStoreBase


def run_rag_pipeline(
    question: str,
    vector_store: VectorStoreBase,
    llm: ChatAnthropic,
    chat_history: Optional[list[dict[str, str]]] = None,
) -> tuple[str, list]:

    k = int(os.getenv("TOP_K_RESULTS", "4"))

    sources = vector_store.similarity_search(question, k)

    answer = generate_answer(question, sources, llm, chat_history)

    return answer, sources
