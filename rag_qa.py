


""" Chatbot Q&A """

import os
import google.generativeai as genai
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings


genai.configure(api_key="AIzaSyCYb5X4_DumVFHWAx3r0fMIBi6rzdG2NZ0")


PROMPT_TEMPLATE = """
You are an AI assistant helping recruiters identify qualified candidates.

Based on the following CV excerpts, answer the question:

Context:
{context}

Question:
{question}

Only use the provided context. If no answer is found, respond with "No one found."

Answer:
"""


def load_vectorstore(path="./chroma_store"):
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = Chroma(persist_directory=path, embedding_function=embedding_model)
    return vectorstore


def ask_question(question: str, k: int = 3) -> str:
    vectorstore = load_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})

    docs = retriever.invoke(question)
    context = "\n\n".join([doc.page_content for doc in docs])

    prompt = PROMPT_TEMPLATE.format(context=context, question=question)

    model = genai.GenerativeModel("models/gemini-1.5-flash")
    response = model.generate_content(prompt)

    return response.text.strip()