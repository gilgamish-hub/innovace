# resume_rater.py

import os
import traceback
import google.generativeai as genai

# -------------------------------
# Safe Gemini call wrapper
# -------------------------------
def ask_gemini_safe(client, prompt, model="gemini-2.0-flash"):
    try:
        gen_model = client.GenerativeModel(model)
        resp = gen_model.generate_content(prompt)

        if hasattr(resp, "text") and resp.text:
            return resp.text.strip(), None
        if hasattr(resp, "candidates") and resp.candidates:
            parts = resp.candidates[0].content.parts
            txt = " ".join([p.text for p in parts if hasattr(p, "text")])
            return txt.strip() or "⚠️ No text returned", None
        return str(resp), None
    except Exception as e:
        return None, str(e) + "\n" + traceback.format_exc()

# -------------------------------
# Resume rating function
# -------------------------------
def rate_resume(resume_text, internship_domain="software engineering"):
    """
    Takes resume text and domain (e.g. "data science") 
    and returns a selection score (%) with insights.
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("⚠️ No GEMINI_API_KEY found. Please set it in your environment.")

    genai.configure(api_key=api_key)
    client = genai

    prompt = f"""
    You are an internship recruiter AI. 
    Evaluate the following resume for a {internship_domain} internship.
    
    Resume:
    {resume_text}
    
    Please output:
    1. A selection score between 0–100% (resume strength).
    2. 2-3 sentences explaining strengths.
    3. 2-3 suggestions to improve chances of selection.
    """

    result, err = ask_gemini_safe(client, prompt)
    if err:
        return {"error": err}
    return {"resume_evaluation": result}

# -------------------------------
# Example usage
# -------------------------------
if __name__ == "__main__":
    # Example dummy resume
    resume_text = """
    Jatin Pal
    Skills: Python, Machine Learning, Deep Learning, Data Analysis
    Experience: Built CNN model for plant disease detection, 
                internship in data science.
    Education: B.Tech CSE 2023–2027
    """

    evaluation = rate_resume(resume_text, internship_domain="machine learning")
    print("\n=== Resume Evaluation ===")
    print(evaluation)
