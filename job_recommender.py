

""" Match each candidate to their top jobs """

from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
import google.generativeai as genai
import re
import numpy as np

job_list = [
    {"title": "AI Research Intern", "description": "Deep learning, NLP, PyTorch, and academic research experience preferred."},
    {"title": "Computer Vision Engineer", "description": "Build image recognition using CNNs, OpenCV, and Python."},
    {"title": "Data Analyst (Tableau)", "description": "Strong Tableau, SQL, Excel, statistics background needed."},
    {"title": "NLP Engineer", "description": "Build language models, transformers, and pipelines for text classification."},
    {"title": "Machine Learning Engineer", "description": "Develop end-to-end ML pipelines using Python, Scikit-learn, TensorFlow."},
    {"title": "Frontend Developer", "description": "React, TypeScript, Tailwind, and UI testing experience required."},
    {"title": "Backend Developer", "description": "Develop APIs using Django or FastAPI, PostgreSQL, Redis."},
    {"title": "Data Scientist", "description": "Data modeling, predictive analytics, A/B testing, and feature engineering."},
    {"title": "Generative AI Engineer", "description": "Work with LLMs, prompt engineering, RAG pipelines, and vector DBs."},
    {"title": "Cloud DevOps Engineer", "description": "Set up CI/CD, Docker, Kubernetes, and deploy ML systems on AWS."},
    {"title": "Cybersecurity Analyst", "description": "Monitor systems for threats, use SIEM tools, and perform vulnerability analysis."},
    {"title": "Business Intelligence Developer", "description": "Build dashboards using Power BI, DAX, and data warehousing skills."},
    {"title": "AI Product Manager", "description": "Define product roadmaps for ML systems and coordinate with cross-functional teams."},
    {"title": "Mobile App Developer", "description": "Develop Flutter or Android apps and integrate REST APIs."},
    {"title": "Software QA Engineer", "description": "Write unit tests, automate UI testing, and use tools like Selenium."},
    {"title": "Bioinformatics Researcher", "description": "Analyze DNA data using ML, sequence modeling, and biomedical NLP."},
    {"title": "Computer Science Instructor", "description": "Teach algorithms, data structures, Python, and ML to students."},
    {"title": "AI for Healthcare Specialist", "description": "Apply ML to medical data, detect anomalies, and support diagnosis."},
    {"title": "Robotics Engineer", "description": "Design intelligent robotic systems using sensor fusion and control theory."},
    {"title": "Speech Recognition Scientist", "description": "Build models for audio transcription, speech-to-text using transformers."}
]



embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


def explain_match_with_llm(cv_text, job_title, job_description):
    prompt = f"""
You are an AI career assistant.

Candidate CV:
\"\"\"{cv_text}\"\"\"

Job Title:
{job_title}

Job Description:
\"\"\"{job_description}\"\"\"

Does this candidate seem like a good fit? If yes, explain why based on their skills and experience in 2–3 sentences.
"""

    try:
        model = genai.GenerativeModel("models/gemini-1.5-flash")
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"⚠️ Gemini Error: {e}"


def recommend_jobs_for_candidate(cv_text, job_list, top_k=3):

    cv_embedding = embedding_model.embed_documents([cv_text])[0]

    job_scores = []

    for job in job_list:
        job_text = f"{job['title']}: {job['description']}"
        job_embedding = embedding_model.embed_documents([job_text])[0]


        score = cosine_similarity(cv_embedding, job_embedding)
        job_scores.append((job, score))


    top_matches = sorted(job_scores, key=lambda x: x[1], reverse=True)[:top_k]


    print("\n🔍 Top Job Recommendations for Candidate:\n")
    for rank, (job, score) in enumerate(top_matches, 1):
        print(f"{rank}. 💼 {job['title']}  —  Similarity Score: {score:.2f}")
        reason = explain_match_with_llm(cv_text, job['title'], job['description'])
        print(f"   🤖 Gemini Reasoning: {reason}\n")


def cosine_similarity(vec1, vec2):
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))