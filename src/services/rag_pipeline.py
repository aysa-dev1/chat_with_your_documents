from src.generation.llm import generate_answer
from src.retrieval.vector_store import VectorStoreBase
from langchain_anthropic import ChatAnthropic
import os

def run_rag_pipeline(question: str, vector_store: VectorStoreBase, llm: ChatAnthropic) -> str:
    
    k = int(os.getenv("TOP_K_RESULTS", "4"))

    docs = vector_store.similarity_search(question, k)

    answer = generate_answer(question, docs, llm)

    return answer