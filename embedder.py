

""" Preprocess and embed CVs into vector store """

import os
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain.docstore.document import Document

VECTOR_STORE_PATH = "./chroma_store"

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

splitter = CharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separator="\n"
)


def embed_chunked_cvs(cv_texts: list, filenames: list, persist=True):
    all_docs = []

    for text, fname in zip(cv_texts, filenames):
        chunks = splitter.split_text(text)
        name = os.path.basename(fname)

        docs = [Document(page_content=chunk, metadata={"name": name}) for chunk in chunks]
        all_docs.extend(docs)

    if os.path.exists(VECTOR_STORE_PATH):
        print("Loading existing vectorstore...")
        vectorstore = Chroma(collection_name="cv_store", embedding_function=embedding_model)
        vectorstore.add_documents(all_docs)
    else:
        print("Creating new vectorstore...")
        vectorstore = Chroma.from_documents(all_docs, embedding_model, persist_directory=VECTOR_STORE_PATH)

    if persist:
        vectorstore.persist()

    return vectorstore


def load_existing_vectorstore():
    return Chroma(persist_directory=VECTOR_STORE_PATH, embedding_function=embedding_model)
