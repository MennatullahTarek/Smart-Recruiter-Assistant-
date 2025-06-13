

"""Generate a 3–4 line summary per CV"""

import google.generativeai as genai


def summarize_cv(cv_text: str) -> str:
    prompt = f"""
You are an AI assistant helping a recruiter. Summarize this candidate's CV in 3–4 lines.

The summary MUST include:
- Key technical and soft skills
- Most recent or relevant roles/projects
- Mention of experience years (if found)
- Education background

Text:
\"\"\"{cv_text}\"\"\"

Return a concise professional summary.
"""

    try:
        model = genai.GenerativeModel("models/gemini-1.5-flash")
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"⚠️ Gemini Error: {e}"