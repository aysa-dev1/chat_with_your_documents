from src.generation.llm import generate_answer
from src.retrieval.vector_store import VectorStoreBase
from langchain_anthropic import ChatAnthropic
import os

def run_rag_pipeline(question: str, vector_store: VectorStoreBase, llm: ChatAnthropic) -> tuple[str, list]:
    
    k = int(os.getenv("TOP_K_RESULTS", "4"))

    sources = vector_store.similarity_search(question, k)

    answer = generate_answer(question, sources, llm)

    return answer, sources