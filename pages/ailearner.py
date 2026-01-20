import streamlit as st
import google.generativeai as genai
import requests
import os

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="AI Learning Path Generator",
    page_icon="🎓",
    layout="centered"
)

# ================= API KEYS =================
GOOGLE_API_KEY = st.secrets["KEY_1"]
YOUTUBE_API_KEY = st.secrets["youtube"]
GITHUB_TOKEN = st.secrets.get("GITHUB_TOKEN", None)

genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel("gemini-2.5-flash-lite")

# ================= SESSION STATE =================
if "learning_plan" not in st.session_state:
    st.session_state.learning_plan = ""

if "history" not in st.session_state:
    st.session_state.history = []

# ================= API HELPERS =================
def search_youtube(query, max_results=5):
    url = "https://www.googleapis.com/youtube/v3/search"
    params = {
        "part": "snippet",
        "q": query,
        "key": YOUTUBE_API_KEY,
        "maxResults": max_results,
        "type": "video",
        "relevanceLanguage": "en"
    }

    response = requests.get(url, params=params)
    if response.status_code != 200:
        return []

    videos = []
    for item in response.json().get("items", []):
        video_id = item["id"]["videoId"]
        title = item["snippet"]["title"]
        videos.append((title, f"https://www.youtube.com/watch?v={video_id}"))

    return videos


def search_github(query, max_results=5):
    url = "https://api.github.com/search/repositories"

    headers = {}
    if GITHUB_TOKEN:
        headers["Authorization"] = f"token {GITHUB_TOKEN}"

    params = {
        "q": query,
        "sort": "stars",
        "order": "desc",
        "per_page": max_results
    }

    response = requests.get(url, headers=headers, params=params)

    if response.status_code == 403:
        st.warning("⚠️ GitHub API rate limit reached. Try again later.")
        return []

    if response.status_code != 200:
        return []

    repos = []
    for item in response.json().get("items", []):
        repos.append((
            item["full_name"],
            item["html_url"],
            item.get("description", "No description provided")
        ))

    return repos


def generate_learning_plan(context):
    prompt = f"""
You are an expert learning coach.

Create a personalized learning plan including:
- Weekly roadmap
- Daily tasks (30–90 minutes)
- Topics to search on YouTube
- Hands-on project ideas

User context:
{context}

Format clearly using markdown.
"""
    response = model.generate_content(prompt)
    return response.text

# ================= UI =================
st.title("🎓 AI-Personalized Learning Path Generator")
st.caption("Gemini • YouTube • GitHub (Hybrid Mode)")

st.divider()

# ================= USER INPUT =================
with st.form("onboarding"):
    goal = st.text_input("🎯 Learning Goal", placeholder="Become a backend developer")
    level = st.selectbox("📊 Current Level", ["Beginner", "Intermediate", "Advanced"])
    time_per_day = st.slider("⏱️ Daily Time (minutes)", 30, 180, 60)
    duration = st.selectbox("📆 Target Duration", ["1 Month", "3 Months", "6 Months"])
    style = st.multiselect(
        "🎧 Preferred Learning Style",
        ["Videos", "Articles", "Hands-on Projects"],
        default=["Videos", "Hands-on Projects"]
    )

    submitted = st.form_submit_button("🚀 Generate Learning Plan")

# ================= PLAN GENERATION =================
if submitted and goal:
    context = f"""
Goal: {goal}
Level: {level}
Time per day: {time_per_day} minutes
Duration: {duration}
Learning style: {', '.join(style)}
"""

    with st.spinner("🧠 Generating your personalized learning plan..."):
        plan = generate_learning_plan(context)

    st.session_state.learning_plan = plan
    st.session_state.history.append(plan)

# ================= DISPLAY PLAN =================
if st.session_state.learning_plan:
    st.subheader("📘 Your Learning Plan")
    st.markdown(st.session_state.learning_plan)

    st.divider()

    # ================= RESOURCE ENRICHMENT =================
    st.subheader("📺 Recommended YouTube Videos")
    for title, link in search_youtube(goal):
        st.markdown(f"- [{title}]({link})")

    st.subheader("💻 Recommended GitHub Projects")
    for name, link, desc in search_github(goal):
        st.markdown(f"- **[{name}]({link})** — {desc}")

    st.divider()

    # ================= FEEDBACK LOOP =================
    st.subheader("🔁 Daily Feedback")
    difficulty = st.slider("Difficulty", 1, 5, 3)
    confidence = st.slider("Confidence", 1, 5, 3)
    notes = st.text_area("Notes / blockers")

    if st.button("♻️ Adapt Learning Plan"):
        feedback_context = f"""
Original Plan:
{st.session_state.learning_plan}

User Feedback:
Difficulty: {difficulty}
Confidence: {confidence}
Notes: {notes}

Adapt the remaining plan accordingly.
"""
        with st.spinner("🔄 Updating plan..."):
            updated_plan = generate_learning_plan(feedback_context)

        st.session_state.learning_plan = updated_plan
        st.session_state.history.append(updated_plan)
        st.success("✅ Plan updated based on your feedback")

# ================= HISTORY =================
with st.expander("🗂️ Learning Plan History"):
    for i, version in enumerate(st.session_state.history):
        st.markdown(f"### Version {i+1}")
        st.markdown(version)
        st.divider()

