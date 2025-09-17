import streamlit as st

st.set_page_config(
    page_title="My App",
    page_icon="🌐",
    initial_sidebar_state="expanded"
)

# --- CSS to hide absolutely everything Streamlit adds ---
hide_streamlit_style = """
<style>
#MainMenu {visibility: hidden;} /* Hides hamburger menu */
header {visibility: hidden;} /* Hides entire header */
footer {visibility: hidden;} /* Hides entire footer */
[data-testid="stStatusWidget"] {display: none;} /* Hides 'Running' status */
[data-testid="stToolbar"] {display: none;} /* Hides toolbar (Deploy button etc.) */
a {display: none !important;} /* Hides all links */
</style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# ---- Home Page ----
st.title("🌐 Welcome to My App")
st.write("Use the sidebar to navigate to other pages.")

# ✅ Sidebar content for Home
st.sidebar.title("📌 Navigation")
st.sidebar.info("👉 Use the selector above to switch pages.\n\nYou’re currently on **Home**.")