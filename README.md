# 🤖 Smart Recruiter Assistant

An intelligent, multi-agent AI assistant that helps recruiters extract insights from CVs, match candidates to job descriptions, summarize profiles, and recommend job roles — all powered by Retrieval-Augmented Generation (RAG), vector search, and Gemini AI.

👉 **Live App**: [Try it on Streamlit](https://e27t3zbdcqlgfd3ktkehx4.streamlit.app/)  

---

## 🚀 Features

✅ Upload multiple CVs (PDF/DOCX)  
✅ Ask natural language questions (e.g., “Who has experience in NLP?”)  
✅ Match candidates to job descriptions  
✅ Generate 3–4 line CV summaries  
✅ Recommend jobs to each candidate  
✅ Gemini-powered LLM reasoning for explainability

---

## 📂 Project Structure

```
Smart-Recruiter-Assistant/
├── app/                    # Backend logic
│   ├── __init__.py
│   ├── config.py           # Central constants (models, paths)
│   ├── parser.py           # Extract text from PDF/DOCX
│   ├── embedder.py         # Embed CVs to vectorstore
│   ├── matcher.py          # Job → candidate matching
│   ├── rag_qa.py           # RAG-based Q&A with Gemini
│   ├── summarizer.py       # CV summary generator
│   ├── job_recommender.py  # Candidate → job recommendations
├── ui/
│   └── main.py    # Streamlit frontend
├── uploaded_files/         # CVs uploaded during runtime
├── requirements.txt
├── README.md
└── .gitignore
```

---

## 🛠 Installation

1. **Clone the repository**:

```bash
git clone https://github.com/MennatullahTarek/Smart-Recruiter-Assistant-.git
cd Smart-Recruiter-Assistant-
```

2. **Install dependencies**:

```bash
pip install -r requirements.txt
```

3. **Run the app locally**:

```bash
streamlit run ui/main.py
```

> 📌 Recommended: Use Python ≤ 3.11  
> ⚠️ Python 3.13+ may break due to sqlite or PyTorch issues.

---

## 💡 Built With

- [LangChain](https://python.langchain.com/)
- [ChromaDB](https://docs.trychroma.com/)
- [HuggingFace Transformers](https://huggingface.co/)
- [Google Gemini API](https://makersuite.google.com/)
- [Streamlit](https://streamlit.io/)



---

## 📄 License

This project is licensed under the MIT License — feel free to fork, modify, and build your own version!
