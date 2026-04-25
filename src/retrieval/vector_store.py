import os
from abc import ABC, abstractmethod

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings


class VectorStoreBase(ABC):
    @abstractmethod
    def add_documents(self, documents: list[Document]) -> None:
        pass

    @abstractmethod
    def similarity_search(self, query: str, k: int) -> list[Document]:
        pass


class ChromaVectorStore(VectorStoreBase):

    def __init__(self, embedder: HuggingFaceEmbeddings, collection_name: str):
        persist_directory = os.getenv("CHROMA_PERSIST_DIR", "./data/chroma")

        self._store = Chroma(
            collection_name=collection_name,
            embedding_function=embedder,
            persist_directory=persist_directory,
        )

    def add_documents(self, documents: list[Document]) -> None:
        if not documents:
            return
        self._store.add_documents(documents)

    def similarity_search(self, query: str, k: int) -> list[Document]:
        return self._store.similarity_search(query, k=k)
