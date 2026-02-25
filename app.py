import streamlit as st
from streamlit.components.v1 import html


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


st.set_page_config(
    page_title="My App",
    page_icon="🌐",
    initial_sidebar_state="expanded"
)

# ---- Home Page ----
st.title("🌐 Welcome to My App")





st.link_button("Go to CSVVISUALISATION", "https://csvvisualisation.streamlit.app")

st.link_button("Go to AUDIOSTORY", "https://exploreai.streamlit.app/audiostory")



st.link_button("Go to SINGIFY", "https://exploreai.streamlit.app/singify")

st.link_button("Go to TEXT2AUDIO", "https://exploreai.streamlit.app/text2audio")



st.link_button("Go to AIPODCAST", "https://exploreai.streamlit.app/aipodcast")

st.link_button("Go to SINGPERFECT", "https://exploreai.streamlit.app/singperfect")

st.link_button("Go to AIPARTNER", "https://exploreai.streamlit.app/aipartner")

st.link_button("Go to MODEL LISTS", "https://ashishraxaul.streamlit.app/list_models")

# ✅ Sidebar content
st.sidebar.title("📌 Navigation")
st.sidebar.info("👉 Use the selector above to switch pages.\n\nYou’re currently on **Home**.")
