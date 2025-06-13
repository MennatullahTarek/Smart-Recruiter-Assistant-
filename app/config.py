import os


EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


COLLECTION_NAME = "cv_store"


GEMINI_MODEL = "models/gemini-1.5-flash"


VECTORSTORE_PATH = os.path.join(os.path.dirname(__file__), "..", "chroma_store")
