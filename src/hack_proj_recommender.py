import pandas as pd

# ---------- Skill Matching Function ----------
def skill_match_ratio(user_skills, internship_skills):
    """
    Returns a match ratio between 0 and 1.
    """
    if not internship_skills:
        return 0
    internship_skills = [s.strip().lower() for s in internship_skills]
    matched = sum(1 for s in user_skills if s.lower() in internship_skills)
    return matched / len(user_skills)

# ---------- Load Dataset ----------
print("=== Hack Project Internship Recommender ===")
dataset_path = input("Enter path to your internship dataset CSV: ").strip()
df = pd.read_csv(dataset_path)
print(f"Dataset loaded with shape: {df.shape}")
print("Columns in dataset:", df.columns.tolist())

# ---------- User Input ----------
user_input = input("Enter your skills separated by commas (e.g., Python,ML,WebDev): ")
user_skills = [skill.strip() for skill in user_input.split(',')]
print("User skills:", user_skills)

# ---------- Preprocess Skills Column ----------
df['skills'] = df['skills'].fillna('')  # replace NaN with empty string
df['skills'] = df['skills'].apply(lambda x: [s.strip().lower() for s in str(x).split(',')])

# ---------- Compute Match Ratio ----------
df['match_ratio'] = df['skills'].apply(lambda x: skill_match_ratio(user_skills, x))

# ---------- Filter Matches ----------
matched_df = df[df['match_ratio'] > 0].sort_values(by='match_ratio', ascending=False)

if matched_df.empty:
    print("No internships match your skills in the dataset.")
else:
    # Show top 10 recommendations
    print(f"Found {len(matched_df)} matching internships. Showing top 10:")
    print(matched_df[['role', 'company_name', 'location', 'skills', 'duration', 'stipend', 'match_ratio']].head(10))
