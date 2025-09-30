import streamlit as st

st.set_page_config(
    page_title="My App",
    page_icon="🌐",
    initial_sidebar_state="expanded"
)

# --- CSS: Hide all unwanted items but KEEP sidebar toggle ---
hide_streamlit_style = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
[data-testid="stStatusWidget"] {display: none;}
[data-testid="stToolbar"] {display: none;}
a[href^="https://github.com"] {display: none !important;}
a[href^="https://streamlit.io"] {display: none !important;}

/* The following specifically targets and hides all child elements of the header's right side,
   while preserving the header itself and, by extension, the sidebar toggle button. */
header > div:nth-child(2) {
    display: none;
}
</style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# ---- Home Page ----
st.title("🌐 Welcome to My App")





st.link_button("Go to CSVVISUALISATION", "https://csvvisualisation.streamlit.app")

st.link_button("Go to AUDIOSTORY", "https://ashishraxaul.streamlit.app/audiostory")

st.link_button("Go to AUDIOSTORY2", "https://ashishraxaul.streamlit.app/audiostory2")

st.link_button("Go to SINGIFY", "https://ashishraxaul.streamlit.app/singify")

st.link_button("Go to MODEL LISTS", "https://ashishraxaul.streamlit.app/list_models")

# ✅ Sidebar content
st.sidebar.title("📌 Navigation")
st.sidebar.info("👉 Use the selector above to switch pages.\n\nYou’re currently on **Home**.")
