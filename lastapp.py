# app_with_enhanced_ui_fixed.py
import os
import traceback
import re
import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
# import google.generativeai as genai
import google.generativeai as genai
# from google import genai
from PyPDF2 import PdfReader
import ollama

from user_input import get_user_profile
from scoring import calculate_selection_score

# -------------------------------
# Enhanced UI Configuration
# -------------------------------
st.set_page_config(
    page_title="InternAI - Smart Internship Finder", 
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# -------------------------------
# Custom CSS for Modern UI
# -------------------------------
def load_custom_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global Styles */
    .main {
        font-family: 'Inter', sans-serif;
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        min-height: 100vh;
    }
    
    /* Header Styling */
    .app-header {
        background: linear-gradient(90deg, #4285F4 0%, #34A853 100%);
        padding: 20px 0;
        margin: -1rem -1rem 2rem -1rem;
        color: white;
        text-align: center;
        box-shadow: 0 4px 20px rgba(66, 133, 244, 0.3);
    }
    
    .app-title {
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    
    .app-subtitle {
        font-size: 1.2rem;
        margin-top: 10px;
        opacity: 0.9;
    }
    
    /* Card Containers */
    .skill-card, .filter-card, .result-card {
        background: white;
        border-radius: 20px;
        padding: 25px;
        margin: 20px 0;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.2);
        backdrop-filter: blur(10px);
    }
    
    .section-header {
        display: flex;
        align-items: center;
        font-size: 1.4rem;
        font-weight: 600;
        margin-bottom: 20px;
        color: #2c3e50;
    }
    
    .section-icon {
        margin-right: 10px;
        font-size: 1.5rem;
    }
    
    /* Stats Cards */
    .stats-container {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
        gap: 20px;
        margin: 30px 0;
    }
    
    .stat-card {
        background: white;
        border-radius: 16px;
        padding: 25px;
        text-align: center;
        box-shadow: 0 6px 25px rgba(0, 0, 0, 0.08);
        border-top: 4px solid #4285F4;
    }
    
    .stat-number {
        font-size: 2.5rem;
        font-weight: 700;
        color: #4285F4;
        margin-bottom: 10px;
    }
    
    .stat-label {
        font-size: 1.1rem;
        color: #6c757d;
        font-weight: 500;
    }
    
    /* Responsive Design */
    @media (max-width: 768px) {
        .app-title { font-size: 2rem; }
        .app-subtitle { font-size: 1rem; }
        .skill-card, .filter-card { padding: 20px; }
        .section-header { font-size: 1.2rem; }
    }
    </style>
    """, unsafe_allow_html=True)

# -------------------------------
# Safe Gemini call wrapper
# -------------------------------
# def ask_gemini_safe(client, prompt, model="gemini-2.5-flash"):
# def ask_gemini_safe(client, prompt, model="gemini-pro"):
#     try:
#         gen_model = client.GenerativeModel(model)
#         resp = gen_model.generate_content(prompt)
#         if hasattr(resp, "text") and resp.text:
#             return resp.text, None
#         if hasattr(resp, "candidates") and resp.candidates:
#             parts = resp.candidates[0].content.parts
#             txt = " ".join([p.text for p in parts if hasattr(p, "text")])
#             return txt or "⚠️ No text returned", None
#         return str(resp), None
#     except Exception as e:
#         return None, str(e) + "\n" + traceback.format_exc()


# def ask_gemini_safe(client, prompt, model="gemini-1.5-pro"):
#     try:
#         response = client.models.generate_content(
#             model=model,
#             contents=prompt
#         )
#         return response.text, None
#     except Exception as e:
#         return None, str(e) + "\n" + traceback.format_exc()

def ask_gemini_safe(client, prompt, model="models/gemini-1.5-flash"):
    try:
        gen_model = client.GenerativeModel(model)
        response = gen_model.generate_content(prompt)

        if hasattr(response, "text") and response.text:
            return response.text, None

        return "⚠️ No response generated", None

    except Exception as e:
        return None, str(e)




def ask_llama(prompt, model="llama3:8b"):
    try:
        response = ollama.chat(
            model=model,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        return response['message']['content'], None
    except Exception as e:
        return None, str(e)
    


def ask_ai(client, prompt):
    # Try Gemini first (if working)
    if client:
        res, err = ask_gemini_safe(client, prompt)
        if res:
            return res, None

    # Fallback to Llama
    res, err = ask_llama(prompt)
    if res:
        return res, None

    return None, "Both Gemini and Llama failed"

# if st.button("Test Gemini"):
#     res, err = ask_gemini_safe(client, "Say hello in one line")

#     if res:
#         st.success(res)
#     else:
#         st.error(err)


# -------------------------------
# Load dataset & build vectorizers
# -------------------------------
DATA_PATH = r"E:\internship recommendation project\data\processed\internship_data_clean.csv"

@st.cache_data
def load_data(path):
    try:
        df = pd.read_csv(path)
        df["skills_clean"] = df.get("skills", "").fillna("").astype(str).str.lower().str.replace(r"[^a-z0-9, ]", "", regex=True)
        df["combined_text"] = df.get("role", "").astype(str) + " " + df.get("company_name", "").astype(str) + " " + df["skills_clean"]
        if "stipend_numeric" not in df.columns:
            df["stipend_numeric"] = (
                df.get("stipend", "").astype(str).str.replace(r"[^\d]", " ", regex=True)
                .str.split().apply(lambda parts: int(parts[0]) if parts and parts[0].isdigit() else 0)
            )
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()

@st.cache_data
def build_vectorizers(df):
    if df.empty:
        return None, None, None, None
    skills_vectorizer = TfidfVectorizer(stop_words="english")
    skills_tfidf = skills_vectorizer.fit_transform(df["skills_clean"])
    text_vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
    text_tfidf = text_vectorizer.fit_transform(df["combined_text"])
    return skills_vectorizer, skills_tfidf, text_vectorizer, text_tfidf

# Load CSS
load_custom_css()

# Load data
df = load_data(DATA_PATH)
if not df.empty:
    skills_vectorizer, skills_tfidf, text_vectorizer, text_tfidf = build_vectorizers(df)
else:
    skills_vectorizer, skills_tfidf, text_vectorizer, text_tfidf = None, None, None, None

# -------------------------------
# Header Section
# -------------------------------
def render_header():
    st.markdown("""
    <div class="app-header">
        <h1 class="app-title">🎯 InternAI</h1>
        <p class="app-subtitle">AI-powered internship recommendations tailored for you</p>
    </div>
    """, unsafe_allow_html=True)

# -------------------------------
# Gemini Configuration
# -------------------------------
# def setup_gemini():
#     with st.sidebar:
#         st.header("🤖 AI Configuration")
#         api_key_input = st.text_input("Enter Gemini API key", type="password", help="Optional: For enhanced AI features")
        
#     api_key = api_key_input or os.getenv("GEMINI_API_KEY")
#     if api_key:
#         # genai.configure(api_key=api_key)
#         client = genai.Client(api_key=api_key)
#         client = genai
#         st.sidebar.success("✅ AI features enabled!")
#         return client
    
#     else:
#         client = None
#         st.sidebar.info("💡 Add API key for AI-powered insights")
#         return None

def setup_gemini():
    with st.sidebar:
        st.header("🤖 AI Configuration")
        api_key_input = st.text_input("Enter Gemini API key", type="password")

    api_key = api_key_input or os.getenv("GEMINI_API_KEY")

    if api_key:
        genai.configure(api_key=api_key)
        st.sidebar.success("✅ AI enabled!")
        return genai
    else:
        st.sidebar.warning("Add API key")
        return None

# -------------------------------
# Skills Management
# -------------------------------
def render_skills_section():
    st.markdown("""
    <div class="skill-card">
        <div class="section-header">
            <span class="section-icon">🎯</span>
            Your Skills
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize skills in session state
    if "user_skills" not in st.session_state:
        st.session_state.user_skills = []
    
    col1, col2 = st.columns([3, 1])
    with col1:
        new_skill = st.text_input("Add a skill...", placeholder="e.g., Python, React, Data Analysis", label_visibility="collapsed")
    with col2:
        if st.button("➕ Add", type="primary"):
            if new_skill and new_skill not in st.session_state.user_skills:
                st.session_state.user_skills.append(new_skill)
                st.rerun()
    
    # Display selected skills
    if st.session_state.user_skills:
        st.markdown(f"**SELECTED ({len(st.session_state.user_skills)})**")
        
        cols = st.columns(5)
        for i, skill in enumerate(st.session_state.user_skills):
            with cols[i % 5]:
                if st.button(f"{skill} ❌", key=f"remove_{skill}", help="Click to remove"):
                    st.session_state.user_skills.remove(skill)
                    st.rerun()
    
    # Popular skills
    st.markdown("**POPULAR SKILLS**")
    popular_skills = ["Python", "JavaScript", "React", "Node.js", "SQL", "HTML/CSS", "Git", "Java", "C++", "Machine Learning"]
    
    cols = st.columns(5)
    for i, skill in enumerate(popular_skills):
        with cols[i % 5]:
            if st.button(skill, key=f"popular_{skill}"):
                if skill not in st.session_state.user_skills:
                    st.session_state.user_skills.append(skill)
                    st.rerun()

# -------------------------------
# Advanced Filters
# -------------------------------
def render_filters():
    st.markdown("""
    <div class="filter-card">
        <div class="section-header">
            <span class="section-icon">🔍</span>
            Advanced Filters
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        stipend_range = st.slider(
            "💰 Stipend Range (₹)",
            min_value=0,
            max_value=50000,
            value=(5000, 25000),
            step=1000
        )
        
        location = st.multiselect(
            "📍 Preferred Locations",
            ["Mumbai", "Delhi", "Bangalore", "Pune", "Hyderabad", "Chennai", "Kolkata", "Remote"],
            default=["Remote"]
        )
    
    with col2:
        duration = st.slider(
            "⏱️ Duration (months)",
            min_value=1,
            max_value=12,
            value=(2, 6)
        )
        
        work_type = st.multiselect(
            "💼 Work Type",
            ["Full-time", "Part-time", "Remote", "Hybrid", "On-site"],
            default=["Remote"]
        )
    
    return {
        "stipend_range": stipend_range,
        "locations": location,
        "duration_range": duration,
        "work_type": work_type
    }

def filter_internships(df, user_skills, filters):
    if df.empty:
        return df
    
    df_filtered = df.copy()
    
    # Apply filters
    if filters["locations"]:
        pattern = "|".join(filters["locations"])
        df_filtered = df_filtered[df_filtered["location"].astype(str).str.contains(pattern, case=False, na=False)]
    
    min_s, max_s = filters["stipend_range"]
    df_filtered = df_filtered[df_filtered["stipend_numeric"].between(min_s, max_s)]
    
    min_d, max_d = filters["duration_range"]
    df_filtered["duration_months"] = df_filtered["duration"].astype(str).str.extract(r"(\d+)").fillna(0).astype(int)
    df_filtered = df_filtered[df_filtered["duration_months"].between(min_d, max_d)]
    
    return df_filtered

def recommend_top_internships(filtered_df, user_skills_text, top_n=10):
    if filtered_df.empty or not text_vectorizer:
        return filtered_df
    
    user_vec = text_vectorizer.transform([user_skills_text])
    subset_mat = text_tfidf[filtered_df.index]
    sims = cosine_similarity(user_vec, subset_mat).flatten()
    top_idx_rel = sims.argsort()[-top_n:][::-1]
    top_idx = filtered_df.index[top_idx_rel]
    results = filtered_df.loc[top_idx].copy()
    results["similarity_score"] = sims[top_idx_rel]
    return results

def render_internship_results(internships_df):
    if internships_df.empty:
        st.warning("🔍 No internships found matching your criteria. Try adjusting your filters!")
        return
    
    # Stats - FIXED VERSION
    st.markdown(f"""
    <div class="stats-container">
        <div class="stat-card">
            <div class="stat-number">{len(internships_df)}</div>
            <div class="stat-label">Matching Internships</div>
        </div>
        <div class="stat-card">
            <div class="stat-number">{internships_df['similarity_score'].mean():.0%}</div>
            <div class="stat-label">Average Match Score</div>
        </div>
        <div class="stat-card">
            <div class="stat-number">₹{internships_df['stipend_numeric'].median():,.0f}</div>
            <div class="stat-label">Median Stipend</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<h2 style='text-align: center; margin: 40px 0;'>🎯 Top Recommendations</h2>", unsafe_allow_html=True)
    
    # FIXED: Use Streamlit containers instead of raw HTML
    for idx, row in internships_df.head(10).iterrows():
        match_score = row.get('similarity_score', 0) * 100
        
        # Create a container for each internship
        with st.container():
            # Use columns for layout
            col1, col2 = st.columns([4, 1])
            
            with col1:
                st.subheader(f"🎯 {row['role']}")
                st.write(f"**🏢 Company:** {row['company_name']}")
                
                # Skills section
                st.write("**Required Skills:**")
                skills_list = str(row['skills_clean']).split(',')
                skills_display = ', '.join([skill.strip().title() for skill in skills_list[:6] if skill.strip()])
                st.write(f"_{skills_display}_")
                
                # Tags in columns
                tag_cols = st.columns(3)
                with tag_cols[0]:
                    st.info(f"📍 {row['location']}")
                with tag_cols[1]:
                    st.info(f"⏱️ {row['duration']}")
                with tag_cols[2]:
                    st.info(f"💰 {row['stipend']}")
            
            with col2:
                # Match score badge
                score_color = "#4CAF50" if match_score >= 70 else "#FF9800" if match_score >= 50 else "#F44336"
                st.markdown(f"""
                <div style="
                    background: {score_color};
                    color: white;
                    padding: 15px;
                    border-radius: 15px;
                    text-align: center;
                    margin-top: 10px;
                ">
                    <div style="font-size: 1.5rem; font-weight: bold;">{match_score:.0f}%</div>
                    <div style="font-size: 0.9rem;">Match Score</div>
                </div>
                """, unsafe_allow_html=True)
            
            st.divider()

# -------------------------------
# Navigation System with session state
# -------------------------------
def render_navigation():
    st.sidebar.title("🧭 Navigation")
    
    pages = {
        "Home": "🏠",
        "Skills Analysis": "💡", 
        "Resume Scanner": "📄",
        "Learning Roadmap": "🗺️",
        "Profile": "👤"
    }
    
    # Page selection
    selected_page = st.sidebar.radio(
        "Go to:",
        list(pages.keys()),
        format_func=lambda x: f"{pages[x]} {x}"
    )
    
    return selected_page

# -------------------------------
# Skills Analysis Page
# -------------------------------
def render_skills_analysis_page(client):
    st.title("💡 Skills Gap Analysis")
    
    if "recommendations" not in st.session_state:
        st.warning("🎯 Please generate recommendations first from the Home page!")
        return
    
    recommendations = st.session_state.recommendations
    user_skills = st.session_state.get("user_skills", [])
    
    # Skills gap analysis
    st.subheader("🎯 Missing Skills Analysis")
    
    # Extract skills from top recommendations
    missing_skills = set()
    user_skills_lower = [skill.lower() for skill in user_skills]
    
    for _, row in recommendations.head(10).iterrows():
        job_skills = str(row.get("skills_clean", "")).split(",")
        for skill in job_skills:
            skill = skill.strip().lower()
            if skill and skill not in user_skills_lower and len(skill) > 2:
                missing_skills.add(skill.title())
    
    missing_skills = list(missing_skills)[:15]
    
    if missing_skills:
        st.write("**Skills You Should Learn:**")
        
        # Display missing skills in columns
        cols = st.columns(5)
        for i, skill in enumerate(missing_skills):
            with cols[i % 5]:
                st.warning(skill)
        
        # AI-powered learning advice
        if client:
            st.subheader("🤖 AI Learning Recommendations")
            
            with st.spinner("🧠 Generating personalized learning advice..."):
                prompt = f"""
                User's current skills: {', '.join(user_skills)}
                Missing skills needed for top internships: {', '.join(missing_skills[:10])}
                
                Provide a concise, actionable learning plan with:
                1. Which skills to prioritize (top 3-5)
                2. Best learning resources for each skill
                3. Estimated time to learn each skill
                4. How these skills will improve internship prospects
                
                Keep it practical and motivating.
                """
                
                advice_text, error = ask_ai(client, prompt)
                
                if advice_text:
                    st.success("🎯 Your Personalized Learning Plan")
                    st.write(advice_text)
                else:
                    st.error("❌ Could not generate learning advice. Please try again.")
                    st.code(error)
    else:
        st.success("🎉 Congratulations! You already have all the key skills for your target internships!")

# -------------------------------
# Resume Scanner Page  
# -------------------------------
def render_resume_scanner_page(client):
    st.title("📄 Resume Scanner & ATS Checker")
    
    uploaded_file = st.file_uploader(
        "Choose your resume file",
        type=["pdf", "txt"],
        help="Supported formats: PDF, TXT"
    )
    
    if uploaded_file:
        try:
            # Extract resume text
            if uploaded_file.name.endswith(".pdf"):
                pdf = PdfReader(uploaded_file)
                resume_text = ""
                for page in pdf.pages:
                    resume_text += page.extract_text() or ""
            else:  # txt
                resume_text = uploaded_file.read().decode("utf-8", errors="ignore")
            
            if resume_text.strip():
                st.success(f"✅ Resume uploaded successfully! ({len(resume_text)} characters)")
                
                # Basic ATS analysis
                word_count = len(resume_text.split())
                has_email = "@" in resume_text
                has_phone = any(char.isdigit() for char in resume_text)
                skill_mentions = sum(1 for skill in st.session_state.get("user_skills", []) if skill.lower() in resume_text.lower())
                
                ats_score = min(95, (
                    (30 if 200 <= word_count <= 800 else 15) +
                    (20 if has_email else 0) +
                    (20 if has_phone else 0) +
                    (25 if skill_mentions >= 3 else skill_mentions * 8) +
                    5
                ))
                
                # Display results
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("ATS Score", f"{ats_score}/100")
                    st.metric("Word Count", word_count)
                    st.metric("Skills Found", skill_mentions)
                
                with col2:
                    st.write("**Analysis:**")
                    st.write(f"✅ Contact info: {'Found' if has_email else 'Missing'}")
                    st.write(f"✅ Phone: {'Found' if has_phone else 'Missing'}")
                    st.write(f"✅ Skills matched: {skill_mentions}")
                
                # AI improvement suggestions
                if client:
                    st.subheader("💡 AI Improvement Suggestions")
                    
                    with st.spinner("Analyzing resume..."):
                        prompt = f"""
                        Analyze this resume and provide 5 specific improvement suggestions:
                        Resume: {resume_text[:2000]}
                        Target skills: {', '.join(st.session_state.get('user_skills', []))}
                        """
                        
                        suggestions, error = ask_ai(client, prompt)
                        if suggestions:
                            st.write(suggestions)
            else:
                st.error("Could not extract text from file")
        except Exception as e:
            st.error(f"Error processing file: {e}")

# -------------------------------
# Learning Roadmap Page
# -------------------------------
def render_learning_roadmap_page(client):
    st.title("🗺️ Personalized Learning Roadmap")
    
    if not client:
        st.warning("🤖 Please configure Gemini AI to generate roadmaps!")
        return
    
    # Roadmap configuration
    col1, col2 = st.columns(2)
    
    with col1:
        timeline = st.selectbox("Timeline", ["4 weeks", "8 weeks", "12 weeks", "6 months"])
        focus_area = st.selectbox("Focus Area", ["Technical Skills", "Soft Skills", "Industry Knowledge"])
    
    with col2:
        learning_style = st.selectbox("Learning Style", ["Online Courses", "Hands-on Projects", "Mixed Approach"])
        time_commitment = st.selectbox("Daily Time", ["1-2 hours", "2-4 hours", "4+ hours"])
    
    if st.button("🚀 Generate Roadmap", type="primary"):
        user_skills = st.session_state.get("user_skills", [])
        
        with st.spinner("Creating your roadmap..."):
            prompt = f"""
            Create a {timeline} learning roadmap for someone with these skills: {', '.join(user_skills)}
            Focus: {focus_area}
            Style: {learning_style}  
            Time: {time_commitment} daily
            
            Include week-by-week breakdown, resources, and milestones.
            """
            
            roadmap, error = ask_ai(client, prompt)
            
            if roadmap:
                st.success("🗺️ Your Personalized Roadmap")
                st.write(roadmap)
            else:
                st.error("Could not generate roadmap")
                st.code(error)

# -------------------------------
# Profile Page
# -------------------------------
def render_profile_page():
    st.title("👤 User Profile")
    
    # Display stats
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Skills Added", len(st.session_state.get("user_skills", [])))
    
    with col2:
        recommendations_count = len(st.session_state.get("recommendations", []))
        st.metric("Recommendations", recommendations_count)
    
    with col3:
        st.metric("Profile Score", "85%")
    
    # Show current skills
    st.subheader("Your Skills")
    current_skills = st.session_state.get("user_skills", [])
    if current_skills:
        for skill in current_skills:
            st.write(f"• {skill}")
    else:
        st.write("No skills added yet")
    
    # Settings
    st.subheader("Settings")
    notifications = st.checkbox("Email notifications")
    weekly_digest = st.checkbox("Weekly digest")
    
    if st.button("Save Settings"):
        st.success("Settings saved!")

# -------------------------------
# Main App Logic
# -------------------------------
def main():
    # Initialize session state
    if "user_skills" not in st.session_state:
        st.session_state.user_skills = []
    
    # Render header
    render_header()
    
    # Setup Gemini
    client = setup_gemini()
    if client:
        if st.button("🧪 Test Gemini"):
            res, err = ask_ai(client, "Say hello in one line")

            if res:
                st.success(res)
            else:
                st.error(err)
    
    # Navigation
    current_page = render_navigation()
    
    # Route to different pages
    if current_page == "Home":
        st.title("Find Your Perfect Internship")
        st.write("Add your skills and get AI-powered recommendations")
        
        # Skills section
        render_skills_section()
        
        # Filters section  
        filters = render_filters()
        
        # Get recommendations button
        if st.button("🚀 Get My Recommendations", type="primary"):
            if not st.session_state.user_skills:
                st.error("Please add at least one skill!")
            elif df.empty:
                st.error("Dataset not available")
            else:
                with st.spinner("Finding internships..."):
                    user_skills_text = " ".join(st.session_state.user_skills)
                    filtered_df = filter_internships(df, st.session_state.user_skills, filters)
                    
                    if not filtered_df.empty:
                        recommendations = recommend_top_internships(filtered_df, user_skills_text)
                        st.session_state.recommendations = recommendations
                        render_internship_results(recommendations)
                    else:
                        st.warning("No internships match your criteria")
    
    elif current_page == "Skills Analysis":
        render_skills_analysis_page(client)
    
    elif current_page == "Resume Scanner":
        render_resume_scanner_page(client)
    
    elif current_page == "Learning Roadmap":
        render_learning_roadmap_page(client)
    
    elif current_page == "Profile":
        render_profile_page()




if __name__ == "__main__":
    main()

