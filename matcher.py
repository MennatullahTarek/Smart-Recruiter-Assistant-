

"""Match a job description to top K CVs"""


from collections import defaultdict
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
import re
import google.generativeai as genai


def extract_keywords(text):
    text = text.lower()
    stopwords = {
        "and", "the", "are", "you", "for", "our", "with", "your", "but", "not", "has",
        "was", "this", "that", "they", "will", "who", "all", "can", "have", "from",
        "preferred", "experience", "skilled", "seeking", "engineer", "we", "job", "role"
    }
    keywords = re.findall(r'\b[a-zA-Z]{3,}\b', text)
    return set(kw for kw in keywords if kw not in stopwords)


def explain_with_llm(question: str, snippet: str):
    prompt = f"""
You are helping a recruiter evaluate candidate CVs.

QUESTION:
{question}

CANDIDATE EXCERPT:
\"\"\"{snippet}\"\"\"

Based on this, explain whether this candidate is relevant to the question. Answer in 1–2 lines.
"""
    try:
        model = genai.GenerativeModel("models/gemini-1.5-flash")
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"⚠️ LLM Error: {e}"


def load_vectorstore(path="./chroma_store"):
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return Chroma(collection_name="cv_store", embedding_function=embedding_model)


def match_job_to_cvs(job_description: str, top_k: int = 5, explain=True):
    vectorstore = load_vectorstore()
    results = vectorstore.similarity_search_with_score(job_description, k=top_k)

    jd_keywords = extract_keywords(job_description)
    grouped_chunks = defaultdict(list)


    for doc, score in results:
        name = doc.metadata.get("name", "Unknown")
        grouped_chunks[name].append((doc.page_content, score))

    print(f"\n🔍 Job Match — Top {top_k} Chunks Grouped by Candidate:\n")
    for i, (candidate, chunks) in enumerate(grouped_chunks.items(), 1):
        print(f"{i}. 📄 Candidate: {candidate}")


        combined_text = " ".join([chunk for chunk, _ in chunks]).lower()
        matched_keywords = sorted([kw for kw in jd_keywords if kw in combined_text])

        if not matched_keywords:
            print("   ❌ Skipping — No actual keyword match in this candidate’s text.\n")
            continue

        explanation = ", ".join(matched_keywords)
        print(f"   💡 Matched terms: {explanation}")


        matching_snippets = []
        header_keywords = ["@gmail", "linkedin", "github", "phone", "email", "010", "cv", "profile"]

        for idx, (chunk, score) in enumerate(chunks):
            chunk_lower = chunk.lower()
            if any(kw in chunk_lower for kw in matched_keywords):
                is_header_like = any(hk in chunk_lower for hk in header_keywords)
                is_first_chunk = idx == 0
                penalty = 1 if is_header_like or is_first_chunk else 0
                matching_snippets.append((chunk, score, penalty))

        if matching_snippets:
            snippet = sorted(matching_snippets, key=lambda x: (x[2], x[1]))[0][0]
        else:
            snippet = chunks[0][0]

        print(f"       📄 Snippet: {snippet.strip().replace(chr(10), ' ')[:200]}...")


        llm_reason = explain_with_llm(job_description, snippet)
        print(f"   🤖 LLM Reasoning: {llm_reason}\n")
