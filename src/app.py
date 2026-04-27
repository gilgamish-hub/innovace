# ---------------------------------------------------
# app.py - Internship Recommender Web App
# ---------------------------------------------------

import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import os
import google.generativeai as genai # type: ignore

# configure Gemini with your API key
genai.configure(api_key="AIzaSyDuSAtLaxrX4-0M0PlP2UPZ6d_G15u9ElU")

def ask_gemini(prompt, model='gemini-1.5-flash'):
    model = genai.GenerativeModel(model)
    response = model.generate_content(prompt)
    return response.text

def filter_internships(df, location=None, stipend_range=None, duration_range=None, company_type=None):
    filtered_df = df.copy()
    if location:
        filtered_df = filtered_df[filtered_df['location'].str.contains('|'.join(location), case=False, na=False)]
    if stipend_range:
        min_s, max_s = stipend_range
        filtered_df = filtered_df[filtered_df['stipend_numeric'].between(min_s, max_s)]
    if duration_range:
        min_d, max_d = duration_range
        filtered_df['duration_months'] = filtered_df['duration'].str.extract(r'(\d+)').astype(float)
        filtered_df = filtered_df[filtered_df['duration_months'].between(min_d, max_d)]
    if company_type:
        filtered_df = filtered_df[filtered_df['intern_type'].str.contains('|'.join(company_type), case=False, na=False)]
    return filtered_df

def recommend_internships(user_skills, filtered_df, top_n=5, use_text=True):
    if use_text:
        user_vec = text_vectorizer.transform([user_skills])
        sims = cosine_similarity(user_vec, text_tfidf[filtered_df.index]).flatten()
    else:
        user_vec = skills_vectorizer.transform([user_skills])
        sims = cosine_similarity(user_vec, skills_tfidf[filtered_df.index]).flatten()
    
    top_idx = sims.argsort()[-top_n:][::-1]
    results = filtered_df.iloc[top_idx].copy()
    results['similarity_score'] = sims[top_idx]
    cols = ["role", "company_name", "location", "stipend_numeric", "duration", "skills_clean", "similarity_score"]
    return results[cols]

# ==================== LOAD DATA ====================
DATA_PATH = r"E:\internship recommendation project\data\processed\internship_data_clean.csv"
df = pd.read_csv(DATA_PATH)
df["skills_clean"] = df["skills"].fillna("").str.lower().str.replace(r"[^a-zA-Z0-9, ]", "", regex=True)
df["combined_text"] = df["role"].astype(str) + " " + df["company_name"].astype(str) + " " + df["skills_clean"]

skills_vectorizer = TfidfVectorizer(stop_words='english')
skills_tfidf = skills_vectorizer.fit_transform(df["skills_clean"])

text_vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
text_tfidf = text_vectorizer.fit_transform(df["combined_text"])

# ==================== STREAMLIT UI ====================
st.set_page_config(page_title="Internship Recommender", layout="wide")
st.title("🎯 Internship Recommendation System")

st.sidebar.header("Candidate Profile")
user_skills = st.sidebar.text_input("Enter your skills (comma-separated):")
locations = st.sidebar.text_input("Preferred Locations (comma-separated):")
stipend_min = st.sidebar.number_input("Minimum Stipend", value=0)
stipend_max = st.sidebar.number_input("Maximum Stipend", value=20000)
duration_min = st.sidebar.number_input("Minimum Duration (months)", value=1)
duration_max = st.sidebar.number_input("Maximum Duration (months)", value=6)
company_type = st.sidebar.text_input("Company Type (Full-time/Part-time/Internship)")

if st.sidebar.button("Find Internships"):
    loc_list = [l.strip() for l in locations.split(',') if l.strip()]
    comp_list = [c.strip() for c in company_type.split(',') if c.strip()]

    filtered_df = filter_internships(
        df,
        location=loc_list if loc_list else None,
        stipend_range=(stipend_min, stipend_max),
        duration_range=(duration_min, duration_max),
        company_type=comp_list if comp_list else None
    )

    if filtered_df.empty:
        st.warning("No internships match your filters. Try relaxing some criteria.")
    else:
        top_internships = recommend_internships(user_skills, filtered_df, top_n=5, use_text=True)
        st.subheader("Top Internship Recommendations")
        for idx, row in top_internships.iterrows():
            st.markdown(f"**{row['role']}** at **{row['company_name']}**")
            st.markdown(f"- Location: {row['location']}")
            st.markdown(f"- Duration: {row['duration']}")
            st.markdown(f"- Stipend: ₹{row['stipend_numeric']}")
            st.markdown(f"- Skills Matched: {row['skills_clean']}")
            st.markdown("---")
        
        # Optional: LLM explanation
        if st.checkbox("Show LLM Insights"):
            prompt = f"Explain why these internships are a good fit for a user with skills: {user_skills}\n{top_internships.to_dict(orient='records')}"
            with st.spinner("Generating insights..."):
                llm_response = ask_gemini(prompt)
            st.subheader("💡 LLM Insights")
            st.write(llm_response)



# D:\project hack\src\lastapp.py