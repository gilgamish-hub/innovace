import sqlite3
import pandas as pd # type: ignore

# --- Load CSV ---
df = pd.read_csv("users_dataset.csv")

# --- Create database connection ---
conn = sqlite3.connect("users.db")
cur = conn.cursor()

# --- Create tables ---
cur.executescript("""
DROP TABLE IF EXISTS UserSkills;
DROP TABLE IF EXISTS Skills;
DROP TABLE IF EXISTS Users;

CREATE TABLE Users (
    user_id TEXT PRIMARY KEY,
    preferred_domain TEXT,
    preferred_location TEXT,
    expected_stipend TEXT,
    preferred_duration TEXT,
    experience_level TEXT,
    availability TEXT
);

CREATE TABLE Skills (
    skill_id INTEGER PRIMARY KEY AUTOINCREMENT,
    skill_name TEXT UNIQUE
);

CREATE TABLE UserSkills (
    user_id TEXT,
    skill_id INTEGER,
    PRIMARY KEY (user_id, skill_id),
    FOREIGN KEY (user_id) REFERENCES Users(user_id) ON DELETE CASCADE,
    FOREIGN KEY (skill_id) REFERENCES Skills(skill_id) ON DELETE CASCADE
);
""")

# --- Insert data into Users ---
users_cols = [
    "UserID","PreferredDomain","PreferredLocation",
    "ExpectedStipend","PreferredDuration","ExperienceLevel","Availability"
]
df_users = df[users_cols].rename(columns={
    "UserID":"user_id",
    "PreferredDomain":"preferred_domain",
    "PreferredLocation":"preferred_location",
    "ExpectedStipend":"expected_stipend",
    "PreferredDuration":"preferred_duration",
    "ExperienceLevel":"experience_level",
    "Availability":"availability"
})
df_users.to_sql("Users", conn, if_exists="append", index=False)

# --- Insert Skills & UserSkills ---
for _, row in df.iterrows():
    user_id = row["UserID"]
    skills = [s.strip() for s in str(row["Skills"]).split(",")]
    for skill in skills:
        cur.execute("INSERT OR IGNORE INTO Skills(skill_name) VALUES (?)", (skill,))
        cur.execute("SELECT skill_id FROM Skills WHERE skill_name=?", (skill,))
        skill_id = cur.fetchone()[0]
        cur.execute("INSERT OR IGNORE INTO UserSkills(user_id, skill_id) VALUES (?, ?)",
                    (user_id, skill_id))

conn.commit()
conn.close()
print("✅ Database 'users.db' created with tables: Users, Skills, UserSkills")
