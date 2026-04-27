# user_input.py
import streamlit as st

def get_user_profile():
    """
    Collects and cleans user input from sidebar:
    - Skills
    - Preferred location(s)
    - Stipend range
    - Duration range (months)
    - Company type
    - Domain / field (optional)
    Returns a dictionary of cleaned user inputs.
    """
    st.sidebar.header("Candidate profile / filters")

    # Skills input
    skills_input = st.sidebar.text_input(
        "Your skills (comma-separated)", value="", key="user_skills"
    )
    skills_list = [s.strip().lower() for s in skills_input.split(",") if s.strip()]

    # Preferred locations
    loc_input = st.sidebar.text_input(
        "Preferred locations (comma-separated)", value="", key="locations"
    )
    loc_list = [l.strip().lower() for l in loc_input.split(",") if l.strip()]

    # Stipend range
    min_stipend = st.sidebar.number_input(
        "Min stipend", value=0, step=500, key="min_stipend"
    )
    max_stipend = st.sidebar.number_input(
        "Max stipend", value=20000, step=500, key="max_stipend"
    )

    # Duration range
    min_duration = st.sidebar.number_input(
        "Min duration (months)", value=1, step=1, key="min_duration"
    )
    max_duration = st.sidebar.number_input(
        "Max duration (months)", value=6, step=1, key="max_duration"
    )

    # Company type filter
    company_input = st.sidebar.text_input(
        "Company type filter (comma-separated)", value="", key="company_type"
    )
    company_list = [c.strip().lower() for c in company_input.split(",") if c.strip()]

    # Domain / field (optional)
    domain_input = st.sidebar.text_input(
        "Preferred domain / field (optional)", value="", key="domain"
    )
    domain_clean = domain_input.strip().lower() if domain_input.strip() else None

    user_profile = {
        "skills_list": skills_list,
        "locations": loc_list,
        "stipend_range": (min_stipend, max_stipend),
        "duration_range": (min_duration, max_duration),
        "company_type": company_list if company_list else None,
        "domain": domain_clean,
        "resume_text": "",
        "skills_text": skills_input  # keep original text for TF-IDF
    }

    return user_profile
