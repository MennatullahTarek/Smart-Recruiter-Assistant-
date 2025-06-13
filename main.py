__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import streamlit as st
import os
from parser import extract_text
from embedder import embed_chunked_cvs
from rag_qa import ask_question
from matcher import match_job_to_cvs
from summarizer import summarize_cv
from job_recommender import recommend_jobs_for_candidate

st.set_page_config(page_title="Smart Recruiter Assistant", layout="wide")
st.title("🤖 Smart Recruiter Assistant")
st.write("Upload CVs, ask questions, and match candidates to jobs using AI.")


if "uploaded_cvs" not in st.session_state:
    st.session_state.uploaded_cvs = []


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


st.subheader("📁 Upload CVs")
uploaded_files = st.file_uploader("Upload multiple CVs (PDF/DOCX)", type=["pdf", "docx"], accept_multiple_files=True)

if st.button("🔍 Upload and Analyze"):
    if uploaded_files:
        cv_paths = []
        texts = []
        for file in uploaded_files:
            file_path = os.path.join("uploaded_" + file.name.replace(" ", "_"))
            with open(file_path, "wb") as f:
                f.write(file.getbuffer())
            cv_paths.append(file_path)
            texts.append(extract_text(file_path))
        embed_chunked_cvs(texts, cv_paths)
        st.session_state.uploaded_cvs = cv_paths
        st.success(f"{len(cv_paths)} CV(s) processed successfully.")
    else:
        st.warning("Please upload at least one file.")


tab1, tab2, tab3, tab4 = st.tabs(["❓ Ask Questions", "🎯 Job Matching", "📝 CV Summarizer", "💼 Job Recommender"])

with tab1:
    st.subheader("Ask questions about candidates")
    query = st.text_input("مثلاً: من يحب اليايمه؟")
    if st.button("Ask"):
        if st.session_state.uploaded_cvs:
            answer = ask_question(query)
            st.text_area("Answer", answer, height=150)
        else:
            st.warning("Please upload and analyze CVs first.")

with tab2:
    st.subheader("Match candidates to a job description")
    job_desc = st.text_area("Paste job description here")
    if st.button("Match Candidates"):
        if st.session_state.uploaded_cvs:
            match_job_to_cvs(job_desc, top_k=3)
            st.info("Matching results printed to terminal.")
        else:
            st.warning("Please upload and analyze CVs first.")

with tab3:
    st.subheader("CV Summaries")
    if st.button("Summarize CVs"):
        if st.session_state.uploaded_cvs:
            for path in st.session_state.uploaded_cvs:
                summary = summarize_cv(extract_text(path))
                st.markdown(f"**📄 {os.path.basename(path)}**")
                st.success(summary)
        else:
            st.warning("Please upload and analyze CVs first.")

with tab4:
    st.subheader("Job Recommendations")
    if st.button("Recommend Jobs"):
        if st.session_state.uploaded_cvs:
            for path in st.session_state.uploaded_cvs:
                st.markdown(f"**📄 {os.path.basename(path)}**")
                recommend_jobs_for_candidate(extract_text(path), job_list, top_k=3)
            st.info("Job recommendations printed in terminal.")
        else:
            st.warning("Please upload and analyze CVs first.")
