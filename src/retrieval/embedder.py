from langchain_huggingface import HuggingFaceEmbeddings


def get_embedder() -> HuggingFaceEmbeddings:

    model_name = "sentence-transformers/all-MiniLM-L6-v2"

    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
