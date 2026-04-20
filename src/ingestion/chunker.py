from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os

def chunk_documents(documents: list[Document]) -> list[Document]:

    chunk_size = int(os.getenv("CHUNK_SIZE", "1000"))
    chunk_overlap = int(os.getenv("CHUNK_OVERLAP", "200"))

    splitter = RecursiveCharacterTextSplitter(
        chunk_size = chunk_size,
        chunk_overlap = chunk_overlap,
        add_start_index = True
    )

    return splitter.split_documents(documents=documents)
