from typing import BinaryIO

from langchain_core.documents import Document
from pypdf import PdfReader


def load_pdf(obj_file: BinaryIO) -> list[Document]:
    """
    Load a PDF file and return a list of LangChain Documents, one per page
    """

    documents: list[Document] = []

    reader = PdfReader(obj_file)

    for page_num, page in enumerate(reader.pages):
        page_text = page.extract_text()

        if not page_text or not page_text.strip():
            continue

        doc = Document(
            page_content=page_text,
            metadata={
                "page": page_num + 1,
                "source": getattr(obj_file, "name", "uploaded_file"),
            },
        )

        documents.append(doc)

    return documents
