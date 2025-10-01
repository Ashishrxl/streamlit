import streamlit as st
import google.generativeai as genai
from google import genai as genaii

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
st.title("🔍 Google Generative AI Models Explorer")

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


# API Key Input (secured with password type input)
api_key = st.secrets["GOOGLE_API_KEY"]
client = genaii.Client(api_key=api_key)


if api_key:
    try:
        genai.configure(api_key=api_key)

        st.success("✅ API key configured successfully!")

        with st.spinner("Fetching models..."):
            models = list(genai.list_models())

        if models:
            for m in models:
                with st.expander(f"📌 {m.name}", expanded=False):
                    st.markdown(f"**Display Name:** {getattr(m, 'display_name', 'N/A')}")
                    st.markdown(f"**Description:** {getattr(m, 'description', 'N/A')}")
                    st.markdown(f"**Input Token Limit:** {getattr(m, 'input_token_limit', 'N/A')}")
                    st.markdown(f"**Output Token Limit:** {getattr(m, 'output_token_limit', 'N/A')}")
                    st.markdown(f"**Supported Generation Methods:** {getattr(m, 'supported_generation_methods', 'N/A')}")
        else:
            st.warning("No models found.")

    except Exception as e:
        st.error(f"❌ Error: {e}")
else:
    st.info("👆 Please enter your API key to fetch available models.")

# --- Model Explorer Helper ---
with st.expander("🔍 Check Available Models with this API Key"):
    try:
        models = client.models.list()
        st.success(f"Found {len(models)} models available for this API key")

        data = []
        for m in models:
            data.append({
                "Model Name": getattr(m, "name", ""),
                "Display Name": getattr(m, "display_name", ""),
                "Supports": ", ".join(getattr(m, "supported_generation_methods", []))
            })
        st.dataframe(pd.DataFrame(data))
        st.info("💡 Look for models where 'Supports' includes `generateImage` for image generation.")
    except Exception as e:
        st.error(f"Failed to list models: {e}")

