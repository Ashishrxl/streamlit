import streamlit as st
import google.generativeai as genai
import requests
import json
from streamlit.components.v1 import html

# ================= HIDE STREAMLIT UI =================
html("""
<script>
try {
  const sel = window.top.document.querySelectorAll(
    '[href*="streamlit.io"], [href*="streamlit.app"]'
  );
  sel.forEach(e => e.style.display='none');
} catch(e) {}
</script>
""", height=0)

st.markdown("""
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
[data-testid="stStatusWidget"] {display: none;}
[data-testid="stToolbar"] {display: none;}
</style>
""", unsafe_allow_html=True)

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="AI Learning Path Generator",
    page_icon="🎓",
    layout="centered"
)

# ================= API KEYS =================
GOOGLE_API_KEY = st.secrets["KEY_2"]
YOUTUBE_API_KEY = st.secrets["youtube"]
GITHUB_TOKEN = st.secrets.get("GITHUB_TOKEN", None)

genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel("gemini-2.5-flash-lite")

# ================= SESSION STATE =================
st.session_state.setdefault("learning_plan", "")
st.session_state.setdefault("history", [])
st.session_state.setdefault("resource_decision", {})

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
    try:
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        return [
            (i["snippet"]["title"], f"https://www.youtube.com/watch?v={i['id']['videoId']}")
            for i in r.json().get("items", [])
        ]
    except Exception:
        return []


def search_github(query, max_results=5):
    url = "https://api.github.com/search/repositories"
    headers = {"Accept": "application/vnd.github+json"}
    if GITHUB_TOKEN:
        headers["Authorization"] = f"token {GITHUB_TOKEN}"

    params = {
        "q": query,
        "sort": "stars",
        "order": "desc",
        "per_page": max_results
    }

    try:
        r = requests.get(url, headers=headers, params=params, timeout=10)
        r.raise_for_status()
        data = r.json()
    except Exception:
        return []

    if "items" not in data or not isinstance(data["items"], list):
        return []

    repos = []
    for item in data["items"]:
        if isinstance(item, dict) and "html_url" in item:
            repos.append({
                "name": item.get("full_name", "Unknown"),
                "url": item["html_url"],
                "description": item.get("description") or "No description available"
            })

    return repos


# ================= AI LOGIC =================
def decide_resources(goal, style):
    prompt = f"""
You are an education strategist.

Goal: {goal}
Learning style: {', '.join(style)}

Decide required resources.
Return ONLY valid JSON:

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
        return {
            "use_github": "Hands-on Projects" in style,
            "use_case_studies": True,
            "use_practice": True,
            "use_reading_guides": True
        }


def generate_learning_plan(context):
    prompt = f"""
You are an expert learning coach.

Create a personalized learning plan with:
- Weekly roadmap
- Daily tasks (30–90 minutes)
- Topics to search on YouTube
- Practice or projects if relevant

Context:
{context}

Use markdown formatting.
"""
    return model.generate_content(prompt).text


def simple_llm(prompt):
    return model.generate_content(prompt).text


# ================= UI =================
st.title("🎓 AI-Personalized Learning Path Generator")
st.caption("Gemini • YouTube • Adaptive Resources")

st.divider()

# ================= FORM =================
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

# ================= GENERATION =================
if submitted and goal:
    context = f"""
Goal: {goal}
Level: {level}
Time per day: {time_per_day} minutes
Duration: {duration}
Learning style: {', '.join(style)}
"""

    with st.spinner("🧠 Generating learning plan..."):
        st.session_state.learning_plan = generate_learning_plan(context)
        st.session_state.resource_decision = decide_resources(goal, style)
        st.session_state.history.append(st.session_state.learning_plan)

# ================= DISPLAY =================
if st.session_state.learning_plan:
    st.subheader("📘 Your Learning Plan")
    st.markdown(st.session_state.learning_plan)

    st.divider()

    # ---------- YouTube ----------
    st.subheader("📺 Recommended YouTube Videos")
    videos = search_youtube(goal)
    if videos:
        for title, link in videos:
            st.markdown(f"- [{title}]({link})")
    else:
        st.info("No YouTube videos found.")

    st.markdown("---")

    # ---------- GitHub (SAFE) ----------
    if st.session_state.resource_decision.get("use_github"):
        repos = search_github(goal)
        if repos:
            st.subheader("💻 Recommended GitHub Projects")
            for repo in repos:
                st.markdown(
                    f"- **[{repo['name']}]({repo['url']})**  \n"
                    f"  _{repo['description']}_"
                )
        else:
            st.info("ℹ️ No suitable GitHub repositories found.")

    # ---------- Case Studies ----------
    if st.session_state.resource_decision.get("use_case_studies"):
        st.subheader("📚 Case Studies")
        st.markdown(simple_llm(f"Provide 3 short case studies for learning {goal}."))

    # ---------- Practice ----------
    if st.session_state.resource_decision.get("use_practice"):
        st.subheader("🧪 Practice Exercises")
        st.markdown(simple_llm(f"Create 5 practice exercises for {goal}."))

    # ---------- Reading ----------
    if st.session_state.resource_decision.get("use_reading_guides"):
        st.subheader("📖 Reading Guide")
        st.markdown(simple_llm(f"Create a structured reading guide for {goal}."))

    st.divider()

# ================= HISTORY =================
with st.expander("🗂️ Learning Plan History"):
    for i, version in enumerate(st.session_state.history):
        st.markdown(f"### Version {i+1}")
        st.markdown(version)
        st.divider()