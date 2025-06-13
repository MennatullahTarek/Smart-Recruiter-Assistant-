""" Preprocess and embed CVs into vector store (Streamlit-safe version) """

import os
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain.docstore.document import Document

# In-memory collection name (no persist_directory)
COLLECTION_NAME = "cv_store"

# Embedding model
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Text splitter configuration
splitter = CharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separator="\n"
)

# Global vectorstore variable for session use
vectorstore = None


def embed_chunked_cvs(cv_texts: list, filenames: list):
    """
    Embed and store the chunked CVs in-memory with metadata.
    """
    global vectorstore
    all_docs = []

    for text, fname in zip(cv_texts, filenames):
        chunks = splitter.split_text(text)
        name = os.path.basename(fname)

        docs = [Document(page_content=chunk, metadata={"name": name}) for chunk in chunks]
        all_docs.extend(docs)

    print("⚙️ Embedding documents into memory...")
    if vectorstore is None:
        vectorstore = Chroma(collection_name=COLLECTION_NAME, embedding_function=embedding_model)
    vectorstore.add_documents(all_docs)

    return vectorstore


def load_existing_vectorstore():
    """
    Retrieve the in-memory vector store (if initialized).
    """
    global vectorstore
    if vectorstore is None:
        vectorstore = Chroma(collection_name=COLLECTION_NAME, embedding_function=embedding_model)
    return vectorstore
