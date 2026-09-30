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

```text
User input (skills, filters)
        ↓
Recommendation engine (TF-IDF + cosine similarity)
        ↓
Top internships + match score
        ↓
Skill gap analysis
        ↓
LLM (Gemini, with a local Llama 3 model as fallback)
        ↓
Learning advice  |  Resume feedback  |  Roadmap
```

---

## 🚀 How to Run

### 1. Clone the repository
```bash
git clone https://github.com/gilgamish-hub/innovace.git
cd innovace
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. (Optional) Add a Gemini API key
The internship recommendations work without any key. The AI features (learning advice, resume suggestions, roadmap) need either a Gemini API key or a local Ollama model.

Set the key as an environment variable, or paste it into the sidebar of the app:
```bash
# Windows PowerShell
$env:GEMINI_API_KEY = "your_key"

# macOS / Linux
export GEMINI_API_KEY="your_key"
```
The model defaults to `gemini-2.5-flash` and can be changed with the `GEMINI_MODEL` variable. Never commit a key to the repository.

### 4. Run the app
```bash
streamlit run streamlit_app.py
```

### 🧪 Optional: run with a local LLM (Ollama)
```bash
ollama run llama3:8b
```
The app calls Gemini first and falls back to the local Llama 3 model if Gemini is unavailable.

---

## 📁 Project Structure

```text
streamlit_app.py     Streamlit app: recommendations, skill analysis, resume scanner, roadmap
scoring.py           Skill-overlap score between a profile and recommended internships
user_input.py        Sidebar form for profile and filters
data/raw/            Original internship listings
data/processed/      Cleaned listings used by the app
notebooks/           Exploration: dataset setup, filtering, resume checker, recommender, LLM integration
```

The notebooks were run with local absolute paths; update the paths before re-running them.

---

## 📦 Dataset

Internshala internship listings from the Kaggle dataset `vipul143/internshala-internship-dataset`: 6,642 listings, 6,583 after cleaning.

---

## 📈 Future Improvements

- Explainable AI (why this internship matches you)
- Skill gap scoring (numeric)
- Resume vs job description comparison
- Multi-model routing (DeepSeek + Llama)
- Real-time job scraping

---

## 🤝 Contribution

Feel free to contribute by improving:

- UI/UX
- Model accuracy
- Prompt engineering
- Dataset quality

---

## 📌 Conclusion

InternAI is not just a recommendation system — it is a complete AI-driven career assistant that guides users through:

👉 Skill Assessment → Internship Matching → Resume Improvement → Learning Roadmap
