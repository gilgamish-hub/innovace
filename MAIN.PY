import os
import re
import pandas as pd
from google import genai

# ==================== CONFIGURATION ====================
GEMINI_API_KEY = 'AIzaSyCTCgvXMFL8enBd-hX3SfbkZ1_6WYt34mI'  # Replace with your real key
os.environ['GEMINI_API_KEY'] = GEMINI_API_KEY
client = genai.Client()

dataset_path = r"E:\internship recommendation project\data\processed\internship_data_clean.csv"  # Replace with your dataset path

# ==================== FUNCTIONS ====================
def ask_gemini(prompt, model='gemini-2.5-flash'):
    """Send a prompt to Gemini LLM and return response text."""
    response = client.models.generate_content(model=model, contents=prompt)
    return response.text

# ==================== DATA LOADING ====================
df = pd.read_csv(dataset_path)
print(f"Dataset loaded with shape: {df.shape}")

# ==================== USER INPUT ====================
skills_input = 'Python, ML, WebDev'  # Example: replace with user input
location_input = 'Bangalore, Mumbai'
stipend_input = '5000-20000'
duration_input = '1-6 months'
intern_type_input = 'Internship'  # Example: Internship, Full-time, Part-time

# Preprocess user inputs
user_skills = [skill.strip() for skill in skills_input.split(',')]
user_locations = [loc.strip() for loc in location_input.split(',')]
stipend_min, stipend_max = map(int, stipend_input.split('-'))

# Extract numbers from duration input
duration_numbers = re.findall(r'\d+', duration_input)
duration_min, duration_max = map(int, duration_numbers)

print("User Skills:", user_skills)
print("Location Preference:", user_locations)
print("Stipend Range:", stipend_min, "-", stipend_max)
print("Duration Preference:", duration_min, "-", duration_max, "months")
print("Internship Type:", intern_type_input)

# ==================== DATA FILTERING ====================
filtered_df = df.copy()

# 1. Filter by skills
if 'skills' in df.columns:
    skill_mask = filtered_df['skills'].str.contains('|'.join(user_skills), case=False, na=False)
    filtered_df = filtered_df[skill_mask]

# 2. Filter by location
if 'location' in df.columns:
    location_mask = filtered_df['location'].str.contains('|'.join(user_locations), case=False, na=False)
    filtered_df = filtered_df[location_mask]

# 3. Filter by stipend
if 'stipend_numeric' in df.columns:
    stipend_mask = filtered_df['stipend_numeric'].between(stipend_min, stipend_max)
    filtered_df = filtered_df[stipend_mask]

# 4. Filter by duration
if 'duration' in df.columns:
    duration_mask = filtered_df['duration'].str.extract(r'(\d+)')[0].astype(int).between(duration_min, duration_max)
    filtered_df = filtered_df[duration_mask]

# 5. Filter by internship type
if 'intern_type' in df.columns:
    type_mask = filtered_df['intern_type'].str.contains(intern_type_input, case=False, na=False)
    filtered_df = filtered_df[type_mask]

print("Filtered dataset shape:", filtered_df.shape)

# ==================== PREPARE GEMINI PROMPT ====================
columns_to_keep = ['role', 'company_name', 'location', 'skills', 'duration', 'stipend', 'intern_type', 'perks', 'website_link']
columns_to_keep = [col for col in columns_to_keep if col in filtered_df.columns]

dataset_summary = filtered_df[columns_to_keep].head(10).to_dict(orient='records')

prompt = f"""
You are an internship recommendation assistant.
Based on the user preferences:
- Skills: {user_skills}
- Locations: {user_locations}
- Stipend Range: {stipend_min}-{stipend_max}
- Duration: {duration_min}-{duration_max} months
- Internship Type: {intern_type_input}

Suggest the top 5 internships from this filtered dataset:
{dataset_summary}
Provide recommendations in a numbered list, including role, company_name, location, duration, stipend, type, perks, and website.
"""

# ==================== GEMINI QUERY ====================
if filtered_df.shape[0] > 0:
    response = ask_gemini(prompt)
    recommendations = [rec.strip() for rec in response.split('\n') if rec.strip()]
    print("\nTop Internship Recommendations:")
    for i, rec in enumerate(recommendations, 1):
        print(f"{i}. {rec}")
else:
    print("No internships match your preferences in the dataset.")
