import streamlit as st
import google.generativeai as genai
import requests
import os
import json
from streamlit.components.v1 import html

# ================= HIDE STREAMLIT UI =================
html(
    """
    <script>
    try {
      const sel = window.top.document.querySelectorAll('[href*="streamlit.io"], [href*="streamlit.app"]');
      sel.forEach(e => e.style.display='none');
    } catch(e) {}
    </script>
    """,
    height=0
)

st.markdown("""
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
[data-testid="stStatusWidget"] {display: none;}
[data-testid="stToolbar"] {display: none;}
a[href^="https://github.com"] {display: none !important;}
a[href^="https://streamlit.io"] {display: none !important;}
</style>
""", unsafe_allow_html=True)

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

if "resource_decision" not in st.session_state:
    st.session_state.resource_decision = {}

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
    r = requests.get(url, params=params)
    if r.status_code != 200:
        return []
    return [
        (i["snippet"]["title"], f"https://www.youtube.com/watch?v={i['id']['videoId']}")
        for i in r.json().get("items", [])
    ]


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

    r = requests.get(url, headers=headers, params=params)
    if r.status_code != 200:
        return []

    return [
        (i["full_name"], i["html_url"], i.get("description", "No description"))
        for i in r.json().get("items", [])
    ]


# ================= AI INTELLIGENCE =================
def decide_resources(goal, style):
    """
    Gemini decides which resource types are useful.
    """
    prompt = f"""
You are an education strategist.

Goal: {goal}
Learning style: {', '.join(style)}

Decide which resources are useful.
Respond ONLY in valid JSON.

Schema:
{{
  "use_github": true/false,
  "use_case_studies": true/false,
  "use_practice": true/false,
  "use_reading_guides": true/false
}}
"""
    try:
        response = model.generate_content(prompt)
        return json.loads(response.text)
    except Exception:
        # Safe fallback
        return {
            "use_github": "Hands-on Projects" in style,
            "use_case_studies": True,
            "use_practice": True,
            "use_reading_guides": True
        }


def generate_learning_plan(context):
    prompt = f"""
You are an expert learning coach.

Create a personalized learning plan including:
- Weekly roadmap
- Daily tasks (30–90 minutes)
- Topics to search on YouTube
- Practice or project ideas (if applicable)

Context:
{context}

Format clearly using markdown.
"""
    return model.generate_content(prompt).text


def generate_case_studies(goal):
    prompt = f"""
Suggest 5 real-world case studies or scenarios to learn:
{goal}

Each should include:
- Situation
- Task
- Reflection question
"""
    return model.generate_content(prompt).text


def generate_practice_exercises(goal):
    prompt = f"""
Create 5 practical exercises (no coding required unless necessary) for:
{goal}
"""
    return model.generate_content(prompt).text


def generate_reading_guide(goal):
    prompt = f"""
Create a structured reading guide for:
{goal}

Include:
- Key concepts
- What to focus on
- Common mistakes
"""
    return model.generate_content(prompt).text


# ================= UI =================
st.title("🎓 AI-Personalized Learning Path Generator")
st.caption("Gemini • YouTube • Adaptive Resources")

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
        default=["Videos"]
    )

    submitted = st.form_submit_button("🚀 Generate Learning Plan")

# ================= PLAN GENERATION =================
if submitted and goal:
    context = f"""
Goal: {goal}
Level: {level}
Time per day: {time_per_day}
Duration: {duration}
Style: {', '.join(style)}
"""

    with st.spinner("🧠 Generating plan..."):
        st.session_state.learning_plan = generate_learning_plan(context)
        st.session_state.resource_decision = decide_resources(goal, style)
        st.session_state.history.append(st.session_state.learning_plan)

# ================= DISPLAY =================
if st.session_state.learning_plan:
    st.subheader("📘 Your Learning Plan")
    st.markdown(st.session_state.learning_plan)

    st.divider()

    # ================= YOUTUBE =================
    st.subheader("📺 Recommended YouTube Videos")
    for title, link in search_youtube(goal):
        st.markdown(f"- [{title}]({link})")

    # ================= GITHUB (SMART) =================
    if st.session_state.resource_decision.get("use_github"):
        st.subheader("💻 Recommended GitHub Projects")
        for name, link, desc in search_github(goal):
            st.markdown(f"- **[{name}]({link})** — {desc}")

    # ================= CASE STUDIES =================
    if st.session_state.resource_decision.get("use_case_studies"):
        st.subheader("📚 Case Studies")
        st.markdown(generate_case_studies(goal))

    # ================= PRACTICE =================
    if st.session_state.resource_decision.get("use_practice"):
        st.subheader("🧪 Practice Exercises")
        st.markdown(generate_practice_exercises(goal))

    # ================= READING =================
    if st.session_state.resource_decision.get("use_reading_guides"):
        st.subheader("📖 Reading Guide")
        st.markdown(generate_reading_guide(goal))

    st.divider()

# ================= HISTORY =================
with st.expander("🗂️ Learning Plan History"):
    for i, v in enumerate(st.session_state.history):
        st.markdown(f"### Version {i+1}")
        st.markdown(v)
        st.divider()