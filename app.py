import streamlit as st

st.set_page_config(
    page_title="My App",
    page_icon="🌐",
    initial_sidebar_state="expanded"
)

# --- CSS to hide all Streamlit default UI except sidebar & your content ---
hide_streamlit_style = """
<style>
#MainMenu {visibility: hidden;} /* Hide hamburger menu */
header {visibility: hidden;} /* Hide header */
footer {visibility: hidden;} /* Hide footer */
[data-testid="stStatusWidget"] {display: none;} /* Hide 'Running' status */
[data-testid="stToolbar"] {display: none;} /* Hide toolbar (Deploy button etc.) */
a[href^="https://github.com/streamlit"] {display: none !important;} /* Hide GitHub link */
a[href^="https://streamlit.io"] {display: none !important;} /* Hide Streamlit link */
</style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# ---- Home Page ----
st.title("🌐 Welcome to My App")
st.write("Use the sidebar to navigate to other pages.")

# ✅ Sidebar content (remains visible)
st.sidebar.title("📌 Navigation")
st.sidebar.info("👉 Use the selector above to switch pages.\n\nYou’re currently on **Home**.")