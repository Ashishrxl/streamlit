import streamlit as st
import google.generativeai as genai

# Streamlit App
st.set_page_config(page_title="Google Generative AI Models", layout="wide")
st.title("🔍 Google Generative AI Models Explorer")

# API Key Input (secured with password type input)
api_key = st.secrets["GOOGLE_API_KEY"]

if api_key:
    try:
        genai.configure(api_key=api_key)

        st.success("✅ API key configured successfully!")

        with st.spinner("Fetching models..."):
            models = list(genai.list_models())

        if models:
            for m in models:
                with st.expander(f"📌 {m.name}", expanded=False):
                    st.markdown(f"**Description:** {m.description}")
                    st.markdown(f"**Input Methods:** {m.input_methods}")
                    st.markdown(f"**Output Methods:** {m.output_methods}")
                    st.markdown(f"**Supported Generation Methods:** {m.supported_generation_methods}")
        else:
            st.warning("No models found.")

    except Exception as e:
        st.error(f"❌ Error: {e}")
else:
    st.info("👆 Please enter your API key to fetch available models.")