""" Chatbot Q&A with Gemini + Chroma (Streamlit-safe) """

import google.generativeai as genai
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings

# 🔐 Configure Gemini API Key
genai.configure(api_key="AIzaSyBSS6k7_SaSM44aNmUFvjKzeys7y7qWzjM") 

# 🧠 Prompt template for Q&A
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

# ✅ Load in-memory Chroma vectorstore
def load_vectorstore():
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return Chroma(collection_name="cv_store", embedding_function=embedding_model)

# 💬 Ask natural language question against vector DB
def ask_question(question: str, k: int = 3) -> str:
    vectorstore = load_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})

    docs = retriever.invoke(question)
    context = "\n\n".join([doc.page_content for doc in docs])

    prompt = PROMPT_TEMPLATE.format(context=context, question=question)

    try:
        model = genai.GenerativeModel("models/gemini-1.5-flash")
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"⚠️ Gemini Error: {e}"
