import streamlit as st
import google.generativeai as genai
import os
from streamlit.components.v1 import html

# --- Hide Streamlit UI elements ---
html(
  """
  <script>
  try {
    const sel = window.top.document.querySelectorAll('[href*="streamlit.io"], [href*="streamlit.app"]');
    sel.forEach(e => e.style.display='none');
  } catch(e) { console.warn('parent DOM not reachable', e); }
  </script>
  """,
  height=0
)

disable_footer_click = """
    <style>
    footer {pointer-events: none;}
    </style>
"""
st.markdown(disable_footer_click, unsafe_allow_html=True)

hide_streamlit_style = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
[data-testid="stStatusWidget"] {display: none;}
[data-testid="stToolbar"] {display: none;}
a[href^="https://github.com"] {display: none !important;}
a[href^="https://streamlit.io"] {display: none !important;}
header > div:nth-child(2) {
    display: none;
}
</style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# -------------------- CONFIG --------------------
st.set_page_config(
    page_title="AI Learning Path Generator",
    page_icon="🎓",
    layout="centered"
)

# Load API Key
GOOGLE_API_KEY = st.secrets["KEY_1"]
genai.configure(api_key=GOOGLE_API_KEY)

model = genai.GenerativeModel("gemini-2.5-flash-lite")

# -------------------- SESSION STATE --------------------
if "learning_plan" not in st.session_state:
    st.session_state.learning_plan = ""

if "history" not in st.session_state:
    st.session_state.history = []

# -------------------- UI --------------------
st.title("🎓 AI-Personalized Learning Path Generator")
st.caption("Adaptive learning powered by Google Gemini")

st.divider()

# -------------------- USER INPUT --------------------
with st.form("onboarding"):
    goal = st.text_input("🎯 Learning Goal", placeholder="Become a backend developer")
    level = st.selectbox("📊 Current Level", ["Beginner", "Intermediate", "Advanced"])
    time_per_day = st.slider("⏱️ Daily Time (minutes)", 30, 180, 60)
    duration = st.selectbox("📆 Target Duration", ["1 Month", "3 Months", "6 Months"])
    style = st.multiselect(
        "🎧 Preferred Learning Style",
        ["Videos", "Articles", "Hands-on Projects", "Quizzes"],
        default=["Videos", "Hands-on Projects"]
    )

    submitted = st.form_submit_button("🚀 Generate Learning Plan")

# -------------------- AI GENERATION --------------------
def generate_learning_plan(context):
    prompt = f"""
    You are an expert learning coach.

    Create a personalized learning plan with:
    - Weekly roadmap
    - Daily tasks (30–90 minutes)
    - Resources
    - Practice ideas

    User context:
    {context}

    Output format:
    Week 1:
    - Day 1:
    - Day 2:
    """
    response = model.generate_content(prompt)
    return response.text

if submitted and goal:
    context = f"""
    Goal: {goal}
    Level: {level}
    Time per day: {time_per_day} minutes
    Duration: {duration}
    Learning style: {', '.join(style)}
    """

    with st.spinner("🧠 Building your personalized plan..."):
        plan = generate_learning_plan(context)

    st.session_state.learning_plan = plan
    st.session_state.history.append(plan)

# -------------------- DISPLAY PLAN --------------------
if st.session_state.learning_plan:
    st.subheader("📘 Your Learning Path")
    st.markdown(st.session_state.learning_plan)

    st.divider()

    # -------------------- FEEDBACK LOOP --------------------
    st.subheader("🔁 Daily Feedback")

    difficulty = st.slider("How difficult was today?", 1, 5, 3)
    confidence = st.slider("Confidence level", 1, 5, 3)
    notes = st.text_area("Notes / Issues faced")

    if st.button("♻️ Adapt My Plan"):
        feedback_context = f"""
        Original Plan:
        {st.session_state.learning_plan}

        User Feedback:
        Difficulty: {difficulty}
        Confidence: {confidence}
        Notes: {notes}

        Adapt the future learning plan accordingly.
        """

        with st.spinner("🔄 Updating your plan..."):
            updated_plan = generate_learning_plan(feedback_context)

        st.session_state.learning_plan = updated_plan
        st.session_state.history.append(updated_plan)

        st.success("Plan updated based on your feedback!")

# -------------------- HISTORY --------------------
with st.expander("🗂️ Previous Versions"):
    for i, version in enumerate(st.session_state.history):
        st.markdown(f"**Version {i+1}**")
        st.markdown(version)
        st.divider()

# -------------------- FOOTER --------------------
