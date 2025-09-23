import streamlit as st
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI

def setup_page():
    st.set_page_config(page_title="CSV Visualizer with Forecasting (Interactive)", layout="wide")
    st.title("📊 CSV Visualizer with Forecasting (Interactive)")
    hide_streamlit_style = """
    <style>
    #MainMenu, footer {visibility: hidden;}
    footer {display: none !important;}
    header {display: none !important;}
    #MainMenu {display: none !important;}
    [data-testid="stToolbar"] { display: none !important; }
    .st-emotion-cache-1xw8zd0 {display: none !important;}
    [aria-label="View app source"] {display: none !important;}
    a[href^="https://github.com"] {display: none !important;}
    [data-testid="stDecoration"] {display: none !important;}
    [data-testid="stStatusWidget"] {display: none !important;}
    button[title="Menu"] {display: none !important;}
    </style>
    """
    st.markdown(hide_streamlit_style, unsafe_allow_html=True)

def setup_api():
    try:
        genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            temperature=0.1,
            google_api_key=st.secrets["GOOGLE_API_KEY"]
        )
        return llm
    except Exception as e:
        st.error(f"Error configuring Gemini API: {e}. Please ensure GOOGLE_API_KEY is set in your Streamlit secrets.")
        st.stop()