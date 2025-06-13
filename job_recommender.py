""" Match each candidate to their top jobs """

from langchain.embeddings import HuggingFaceEmbeddings
import google.generativeai as genai
import numpy as np

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def cosine_similarity(vec1, vec2):
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))


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


def recommend_jobs_for_candidate(cv_text, job_list, top_k=3, return_output=False):
    cv_embedding = embedding_model.embed_documents([cv_text])[0]

    job_scores = []
    for job in job_list:
        job_text = f"{job['title']}: {job['description']}"
        job_embedding = embedding_model.embed_documents([job_text])[0]
        score = cosine_similarity(cv_embedding, job_embedding)
        job_scores.append((job, score))

    top_matches = sorted(job_scores, key=lambda x: x[1], reverse=True)[:top_k]

    if return_output:
        results = []
        for job, score in top_matches:
            reason = explain_match_with_llm(cv_text, job['title'], job['description'])
            results.append((job, score, reason))
        return results

    print("\n🔍 Top Job Recommendations for Candidate:\n")
    for rank, (job, score) in enumerate(top_matches, 1):
        print(f"{rank}. 💼 {job['title']}  —  Similarity Score: {score:.2f}")
        reason = explain_match_with_llm(cv_text, job['title'], job['description'])
        print(f"   🤖 Gemini Reasoning: {reason}\n")
