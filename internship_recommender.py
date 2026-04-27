# ---------------------------------------------------
# Internship Recommender with Filters + LLM Insights
# ---------------------------------------------------

import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from google import genai

# ==================== CONFIG ====================
GEMINI_API_KEY = 'AIzaSyCTCgvXMFL8enBd-hX3SfbkZ1_6WYt34mI'  # Replace with your real key
os.environ['GEMINI_API_KEY'] = GEMINI_API_KEY
client = genai.Client()

# ==================== FUNCTIONS ====================
def ask_gemini(prompt, model='gemini-2.5-flash'):
    """Send a prompt to Gemini LLM and return the response text."""
    response = client.models.generate_content(model=model, contents=prompt)
    return response.text

# ==================== DATA ====================
DATA_PATH = r"D:\project hack\data\processed\internship_data_clean.csv"
df = pd.read_csv(DATA_PATH)
print(f"Dataset loaded -> {df.shape}")

# Clean and prepare text
df["skills_clean"] = df["skills"].fillna("").str.lower().str.replace(r"[^a-zA-Z0-9, ]", "", regex=True)
df["combined_text"] = df["role"].astype(str) + " " + df["company_name"].astype(str) + " " + df["skills_clean"]

# TF-IDF vectorizers
skills_vectorizer = TfidfVectorizer(stop_words="english")
skills_tfidf = skills_vectorizer.fit_transform(df["skills_clean"])

text_vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
text_tfidf = text_vectorizer.fit_transform(df["combined_text"])

# ==================== FILTER FUNCTION ====================
def filter_internships(df, location=None, stipend_range=None, duration_range=None, company_type=None):
    filtered_df = df.copy()
    
    # Location filter
    if location:
        filtered_df = filtered_df[filtered_df['location'].str.contains('|'.join(location), case=False, na=False)]
    
    # Stipend filter (assumes 'stipend_numeric' exists)
    if stipend_range:
        min_stipend, max_stipend = stipend_range
        filtered_df = filtered_df[filtered_df['stipend_numeric'].between(min_stipend, max_stipend)]
    
    # Duration filter (assumes 'duration' like '3 Months')
    if duration_range:
        min_months, max_months = duration_range
        # Extract numeric months
        filtered_df['duration_months'] = filtered_df['duration'].str.extract(r'(\d+)').astype(float)
        filtered_df = filtered_df[filtered_df['duration_months'].between(min_months, max_months)]
    
    # Company type filter
    if company_type:
        filtered_df = filtered_df[filtered_df['intern_type'].str.contains('|'.join(company_type), case=False, na=False)]
    
    return filtered_df

# ==================== RECOMMENDATION FUNCTION ====================
def recommend_internships(user_skills, top_n=5, filtered_df=None, use_text=True):
    if filtered_df is None:
        filtered_df = df
    
    if use_text:
        user_vec = text_vectorizer.transform([user_skills])
        sims = cosine_similarity(user_vec, text_tfidf[filtered_df.index]).flatten()
    else:
        user_vec = skills_vectorizer.transform([user_skills])
        sims = cosine_similarity(user_vec, skills_tfidf[filtered_df.index]).flatten()
    
    top_idx = sims.argsort()[-top_n:][::-1]
    results = filtered_df.iloc[top_idx].copy()
    results["similarity_score"] = sims[top_idx]

    cols_to_return = ["role", "company_name", "location", "stipend_numeric", "duration", "skills_clean", "similarity_score"]
    return results[cols_to_return]

# ==================== USER INPUT ====================
user_skills = input("Enter your skills separated by commas: ").strip()
location_pref = input("Enter preferred locations separated by commas (or leave blank): ").strip().split(',')
stipend_min = int(input("Enter minimum stipend (or 0): "))
stipend_max = int(input("Enter maximum stipend (or 999999): "))
duration_min = int(input("Enter minimum duration in months (or 0): "))
duration_max = int(input("Enter maximum duration in months (or 24): "))
company_type_input = input("Enter company type (Full-time/Part-time/Internship) separated by commas (or leave blank): ").strip().split(',')

# Clean empty strings
location_pref = [l for l in location_pref if l]
company_type_input = [c for c in company_type_input if c]

# ==================== FILTER DATA ====================
filtered_df = filter_internships(
    df,
    location=location_pref if location_pref else None,
    stipend_range=(stipend_min, stipend_max),
    duration_range=(duration_min, duration_max),
    company_type=company_type_input if company_type_input else None
)

print(f"\nFiltered dataset shape: {filtered_df.shape}")

# ==================== RECOMMEND TOP INTERNSHIPS ====================
top_internships = recommend_internships(user_skills, top_n=5, filtered_df=filtered_df, use_text=True)
print("\n🔎 Top Internship Recommendations (TF-IDF Scores):")
print(top_internships)

# ==================== LLM EXPLANATION ====================
prompt = f"""
You are an internship recommendation assistant.
Here are some top internship options for a user with skills: {user_skills}
{top_internships.to_dict(orient='records')}
Explain in simple language why each internship is a good fit for the user.
"""
llm_response = ask_gemini(prompt)
print("\n💡 LLM Insights:")
print(llm_response)
