import os
import pandas as pd
from google import genai

# ==================== CONFIGURATION ====================
# Set your Gemini API key directly
GEMINI_API_KEY = 'AIzaSyCTCgvXMFL8enBd-hX3SfbkZ1_6WYt34mI'  # Replace with your real key
os.environ['GEMINI_API_KEY'] = GEMINI_API_KEY
client = genai.Client()

# Set your dataset path directly
dataset_path = r'D:\project hack\data\processed\internship_data_clean.csv'  # Replace with your dataset path

# ==================== FUNCTIONS ====================

def ask_gemini(prompt, model='gemini-2.5-flash'):
    """Sends a prompt to Gemini LLM and returns the response text."""
    response = client.models.generate_content(model=model, contents=prompt)
    return response.text

# ==================== DATA LOADING ====================

df = pd.read_csv(dataset_path)
print('Dataset loaded with shape:', df.shape)
print('Columns in dataset:', df.columns.tolist())

# ==================== USER INPUT ====================
# Skills as a comma-separated string
skills_input = 'Python, ML, WebDev'  # Replace with user skills
user_skills = [skill.strip() for skill in skills_input.split(',')]
print('User skills:', user_skills)

# ==================== DATA FILTERING ====================
# Filter for internships that contain any of the user skills in the 'skills' column
if 'skills' in df.columns:
    technical_mask = df['skills'].str.contains('|'.join(user_skills), case=False, na=False)
    df_filtered = df[technical_mask]
else:
    df_filtered = df

print('Filtered dataset shape:', df_filtered.shape)

# ==================== DATA PREPARATION ====================
# Define columns to keep, only existing columns
columns_to_keep = ['role', 'company_name', 'location', 'skills', 'duration', 'stipend']
columns_to_keep = [col for col in columns_to_keep if col in df_filtered.columns]
print('Using columns:', columns_to_keep)

# Prepare dataset summary
dataset_summary = df_filtered[columns_to_keep].head(10).to_dict(orient='records')

# Prepare prompt for Gemini
prompt = f"""
You are an internship recommendation assistant.
Based on the skills {user_skills}, suggest the top 5 internships from this filtered dataset:
{dataset_summary}
Provide the recommendations in a numbered list, including role and company_name.
"""

# ==================== GEMINI QUERY ====================
response = ask_gemini(prompt)

# ==================== FORMAT OUTPUT ====================
recommendations = [rec.strip() for rec in response.split('\n') if rec.strip()]
print('\nTop Internship Recommendations:')
for i, rec in enumerate(recommendations, 1):
    print(f'{i}. {rec}')