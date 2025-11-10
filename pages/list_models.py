import streamlit as st
import google.generativeai as genai



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

# Streamlit App
st.set_page_config(page_title="Google Generative AI Models", layout="wide")
st.title("🔍 Models list of Google API")

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

# --- API Key selection ---
api_keys = {
    "Key 1": st.secrets["KEY_1"],
    "Key 2": st.secrets["KEY_2"], "Key 3": st.secrets["KEY_3"], "Key 4": st.secrets["KEY_4"], "Key 5": st.secrets["KEY_5"], "Key 6": st.secrets["KEY_6"], "Key 7": st.secrets["KEY_7"], "Key 8": st.secrets["KEY_8"], "Key 9": st.secrets["KEY_9"], "Key 10": st.secrets["KEY_10"], "Key 11": st.secrets["KEY_11"]
}
selected_key_name = st.selectbox("Select Key", list(api_keys.keys()))
api_key = api_keys[selected_key_name]


if api_key:
    try:
        genai.configure(api_key=api_key)

        st.success("✅ API key configured successfully!")

        with st.spinner("Fetching models..."):
            models = list(genai.list_models())

        if models:
            for i, m in enumerate(models, start=1):
                
                with st.expander(f"{i} 📌 {m.name}", expanded=False):
                    st.markdown(f"**Display Name:** {getattr(m, 'display_name', 'N/A')}")
                    st.markdown(f"**Description:** {getattr(m, 'description', 'N/A')}")
                    st.markdown(f"**Input Token Limit:** {getattr(m, 'input_token_limit', 'N/A')}")
                    st.markdown(f"**Output Token Limit:** {getattr(m, 'output_token_limit', 'N/A')}")
                    st.markdown(f"**Supported Generation Methods:** {getattr(m, 'supported_generation_methods', 'N/A')}")
                    st.markdown(f"**Version:** {getattr(m, 'version', 'N/A')}")
                    st.markdown(f"**Temperature (Default):** {getattr(m, 'temperature', 'N/A')}")
                    
        else:
            st.warning("No models found.")

    except Exception as e:
        st.error(f"❌ Error: {e}")
else:
    st.info("👆 Please enter your API key to fetch available models.")

