import streamlit as st

st.set_page_config(
    page_title="My App",
    page_icon="🌐",
    initial_sidebar_state="expanded"
)

# --- CSS to hide everything except sidebar ---
hide_streamlit_style = """
<style>
#MainMenu {visibility: hidden;} /* Hide hamburger menu */
header {visibility: hidden;} /* Hide header */
footer {visibility: hidden;} /* Hide footer */
[data-testid="stStatusWidget"] {display: none;} /* Hide 'Running' status */
[data-testid="stToolbar"] {display: none;} /* Hide toolbar (Deploy button etc.) */
a {display: none !important;} /* Hide all links */
section.main > div {display: none;} /* Hide main content area */
</style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# ✅ Sidebar content (only thing visible)
st.sidebar.title("📌 Navigation")
st.sidebar.info("👉 Use the selector above to switch pages.\n\nYou’re currently on **Home**.")