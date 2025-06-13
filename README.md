# 📄 Smart Recruiter Assistant – A RAG-based CV Query and Job Matching System

> 🔍 An AI-powered recruitment assistant that helps recruiters **search**, **summarize**, and **match candidates** to jobs using **Retrieval-Augmented Generation (RAG)** and **Large Language Models (LLMs)**.

## 🚀 Overview

The Smart Recruiter Assistant is a multi-functional AI system that enables recruiters to:
- 🧠 Ask **natural language questions** like “Who has experience in time series?”
- 📌 Match **job descriptions** to the most suitable candidates
- 📊 Recommend **open roles** to candidates based on their profiles
- 📝 Summarize each CV into a short, professional overview

Built with:
- ✅ **LangChain** (for vector search + chunking)
- ✅ **ChromaDB** (for storing CV embeddings)
- ✅ **Gemini 1.5 Flash** (for LLM reasoning & summaries)
- ✅ **HuggingFace MiniLM** (for lightweight embeddings)

## 💡 Core Features

### 🔎 1. CV Q&A Chatbot (RAG)
Ask recruiter-style questions like:
> “Who graduated from Cairo University?”  
> “Who has experience with Generative AI?”

✅ Uses Chroma vector store to retrieve relevant snippets  
✅ Gemini generates contextual, human-like responses

### 🎯 2. Job Description Matching
Input: A job description  
Output: Top K ranked candidates

✅ Ranks based on:
- Skill match  
- Experience level  
- Education relevance  

✅ Gemini explains *why* each candidate is a fit

### 📄 3. Candidate Summarizer
Automatically generates a 3–4 line summary per CV including:
- Key skills
- Recent roles or projects
- Education
- Years of experience (if available)

### 📌 4. Job Recommendations for Candidates
Input: CV  
Output: Top 3 job matches (from a list of 20+ descriptions)

✅ Uses semantic similarity + Gemini explanations  
✅ Helps job seekers find suitable roles

## 📁 Folder Structure

```
📦 smart-recruiter/
├── main.py                  # Entry point: handles full pipeline
├── parser.py                # Extracts text from PDF/DOCX
├── embedder.py              # Embeds CVs into ChromaDB
├── rag_qa.py                # RAG-based CV Q&A
├── matcher.py               # Job description to candidate matching
├── summarizer.py            # CV summarization using Gemini
├── job_recommender.py       # Recommends jobs for candidates
├── chroma_store/            # Persistent vector store (auto-generated)
```

## 🧪 Example Usage

```bash
python main.py
```

It will:
1. Load and embed CVs
2. Answer a recruiter query
3. Match candidates to a job description
4. Summarize each candidate
5. Recommend jobs to each candidate

## 🔧 Dependencies

Install with:

```bash
pip install -r requirements.txt
```

> Make sure you have access to Gemini API (`google.generativeai`) and HuggingFace models.

## 📌 Tech Stack

| Component | Tool |
|----------|------|
| Text Parsing | PyMuPDF, docx2txt |
| Embeddings | HuggingFace `MiniLM-L6` |
| Vector Store | ChromaDB |
| LLM | Gemini 1.5 Flash |
| RAG Framework | LangChain |
| Similarity | FAISS / cosine |
| Explanation | Gemini reasoning prompts |

