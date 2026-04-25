import streamlit as st
from langchain_anthropic import ChatAnthropic
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv

from src.generation.llm import get_llm
from src.retrieval.embedder import get_embedder
from src.ingestion import ingest_document
from src.services.rag_pipeline import run_rag_pipeline
from src.retrieval.vector_store import ChromaVectorStore

# initialization
load_dotenv()

@st.cache_resource
def load_llm() -> ChatAnthropic:
    return get_llm()

@st.cache_resource
def load_embedder() -> HuggingFaceEmbeddings:
    return get_embedder()

@st.cache_resource
def load_vector_store(collection_name: str) -> ChromaVectorStore:
    embedder = load_embedder()
    return ChromaVectorStore(embedder, collection_name=collection_name)

try:
    llm = load_llm()
except ValueError as e:
    st.error(str(e))
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []
if "ingested_file" not in st.session_state:
    st.session_state.ingested_file = None

# handling file upload
uploaded_file = st.sidebar.file_uploader("Upload your PDF document", type=["pdf"])

collection_name = "default"
vector_store = load_vector_store(collection_name)

if uploaded_file is None and st.session_state.ingested_file is not None:                                                 
    st.session_state.ingested_file = None
    st.session_state.messages = []  

if uploaded_file is not None:
    collection_name = "".join(filter(str.isalnum, uploaded_file.name))
    
    vector_store = load_vector_store(collection_name)

    if uploaded_file.name != st.session_state.ingested_file:
        with st.spinner("Processing document..."):
            
            chunk_count = ingest_document(uploaded_file, vector_store)
            if chunk_count == 0:
                st.sidebar.error("No text could be extracted from this PDF. It may be a scanned document")
            else:
                st.session_state.ingested_file = uploaded_file.name
                st.sidebar.success(f"Indexed {chunk_count} chunks!")
                st.session_state.messages = []


# handling chat history
if st.sidebar.button("Clear conversation"):
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# handling new inputs
if prompt := st.chat_input("Ask a question about your document"):
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    if st.session_state.ingested_file is None:
        st.warning("No document uploaded! Please upload a document in the sidebar, before asking questions")
    else:
        with st.chat_message("assistant"):
            with st.spinner("Your document is being analyzed..."):

                answer, sources = run_rag_pipeline(prompt, vector_store, llm)

                st.markdown(answer)

                with st.expander("View sources"):
                    for i, doc in enumerate(sources):
                        source_name = doc.metadata.get("source", "Unknown")
                        page_num = doc.metadata.get("page", "Unknown")

                        st.write(f"Source {i+1}: {source_name}, Page: {page_num}")
                        st.caption(f"\"{doc.page_content[:250]}...\"")
                        st.divider()
                
                st.session_state.messages.append({"role": "assistant", "content": answer})
