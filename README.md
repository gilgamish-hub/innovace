# 🎯 InternAI – Smart Internship Recommendation System

InternAI is an AI-powered internship recommendation system that helps students discover the most suitable internships based on their skills, preferences, and profile. It combines Machine Learning with Large Language Models (LLMs) to provide personalized insights, skill gap analysis, resume evaluation, and learning roadmaps.

---

## 🔍 Features

### 🧠 1. User Input & Preferences
Users can input:
- Technical skills (e.g., Python, React, ML)
- Preferred stipend range (min–max)
- Location (Remote / specific cities)
- Internship duration & work type

This forms the foundation of personalized recommendations.

---

### 🎯 2. Internship Recommendation Engine
The system uses:
- **TF-IDF Vectorization**
- **Cosine Similarity**

to match user skills with internship listings.

#### Output:
- Top recommended internships
- Match Score (%) for each internship

👉 This score represents how closely your skills align with the job requirements.

---

### 📊 3. Skill Gap Analysis
After generating recommendations, the system identifies:

- Missing skills required for top internships
- Most important skills to learn
- Priority-based learning suggestions

👉 Helps users understand *what they lack* and *what to focus on next*.

---

### 🤖 4. AI Learning Recommendations
Using LLM (Llama / Gemini):

- Personalized learning advice
- Suggested resources
- Estimated learning time
- Career impact of each skill

---

### 📄 5. Resume Scanner & ATS Checker
Users can upload their resume (PDF/TXT).

#### System evaluates:
- Word count
- Contact information presence
- Skill matching
- Resume completeness

#### Output:
- ATS Score (out of 100)
- Resume insights

---

### 💡 6. AI Resume Improvement Suggestions
The AI analyzes the resume and provides:
- Specific improvement suggestions
- Missing sections or keywords
- Optimization tips for shortlisting

---

### 🗺️ 7. Personalized Learning Roadmap
Based on:
- User skills
- Missing skills
- Learning preferences

The system generates:
- Week-by-week roadmap
- Tasks & milestones
- Recommended learning path

---

## ⚙️ Tech Stack

- **Frontend**: Streamlit
- **Machine Learning**:
  - TF-IDF Vectorizer
  - Cosine Similarity
- **LLM Integration**:
  - Ollama (Llama 3 / Phi-3)
  - Optional: Google Gemini API
- **Data Processing**: Pandas, NumPy
- **Resume Parsing**: PyPDF2

---

## 🧠 System Architecture
User Input
↓
ML Recommendation Engine (TF-IDF + Cosine Similarity)
↓
Top Internships + Match Score
↓
Skill Gap Analysis
↓
LLM (Llama/Gemini)
↓

Learning Advice
Resume Feedback
Roadmap Generation



---

## 🚀 How to Run

### 1. Clone the repository
```bash
git clone https://github.com/your-username/internai.git
cd internai


## install dependencies

pip install -r requirements.txt

2. Install dependencies
pip install -r requirements.txt
3. Run the app
streamlit run lastapp.py
🧪 Optional: Run with Local LLM (Ollama)

Run Llama 3 model:

ollama run llama3:8b

Or use a faster lightweight model:

ollama run phi3
📈 Future Improvements
Explainable AI (Why this internship matches you)
Skill gap scoring (numeric)
Resume vs job description comparison
Multi-model routing (DeepSeek + Llama)
Real-time job scraping
🤝 Contribution

Feel free to contribute by improving:

UI/UX
Model accuracy
Prompt engineering
Dataset quality
📌 Conclusion

InternAI is not just a recommendation system — it is a complete AI-driven career assistant that guides users through:

👉 Skill Assessment
→ Internship Matching
→ Resume Improvement
→ Learning Roadmap