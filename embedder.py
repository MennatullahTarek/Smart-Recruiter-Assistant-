

""" Preprocess and embed CVs into vector store """


import os
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain.docstore.document import Document


COLLECTION_NAME = "cv_store"

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

splitter = CharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separator="\n"
)

vectorstore = None


def embed_chunked_cvs(cv_texts: list, filenames: list):
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
    global vectorstore
    if vectorstore is None:
        vectorstore = Chroma(collection_name=COLLECTION_NAME, embedding_function=embedding_model)
    return vectorstore

