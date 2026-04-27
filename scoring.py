# scoring.py
# Compute selection probability for internships

def calculate_selection_score(user_profile, top_internships_df):
    """
    user_profile: dict with keys like 'skills' (comma-separated string)
    top_internships_df: recommended internships (DataFrame) with 'skills_clean'
    
    Returns a score in percentage (0-100)
    """
    if top_internships_df.empty:
        return 0.0

    user_skills = set([s.strip().lower() for s in user_profile.get("skills", "").split(",") if s.strip()])
    scores = []

    for _, row in top_internships_df.iterrows():
        job_skills = set([s.strip().lower() for s in str(row.get("skills_clean", "")).split(",") if s.strip()])
        if not job_skills:
            match_ratio = 0.0
        else:
            match_ratio = len(user_skills & job_skills) / len(job_skills)

        # Optional: weight by stipend preference or location (can be added later)
        scores.append(match_ratio)

    avg_score = sum(scores) / len(scores) if scores else 0.0
    return round(avg_score * 100, 2)  # percentage
