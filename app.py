import streamlit as st

st.set_page_config(
    page_title="My App",
    page_icon="🌐",
    initial_sidebar_state="expanded"
)

# --- CSS: Hide only unwanted items, KEEP sidebar toggle ---
hide_streamlit_style = """
<style>
#MainMenu {visibility: hidden;}         /* Hide hamburger menu */
footer {visibility: hidden;}            /* Hide footer */
[data-testid="stStatusWidget"] {display: none;}  /* Hide status */
[data-testid="stToolbar"] {display: none;}       /* Hide deploy/share toolbar */
a[href^="https://github.com"] {display: none !important;}   /* Hide GitHub link */
a[href^="https://streamlit.io"] {display: none !important;} /* Hide Streamlit link */

/* DO NOT hide <header>, only hide its right-side elements */
header [data-testid="stHeaderActionElements"] {display: none;}
</style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# ---- Home Page ----
st.title("🌐 Welcome to My App")
st.write("Use the sidebar to navigate to other pages.")

# ✅ Sidebar content
st.sidebar.title("📌 Navigation")
st.sidebar.info("👉 Use the selector above to switch pages.\n\nYou’re currently on **Home**.")