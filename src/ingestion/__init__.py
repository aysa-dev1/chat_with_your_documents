from typing import BinaryIO

from src.ingestion.loader import load_pdf
from src.ingestion.chunker import chunk_documents
from src.retrieval.vector_store import VectorStoreBase

def ingest_document(file: BinaryIO, vector_store: VectorStoreBase) -> int:
    documents = load_pdf(file)
    chunks = chunk_documents(documents)
    vector_store.add_documents(chunks)

    return len(chunks)
