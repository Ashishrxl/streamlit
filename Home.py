import streamlit as st
from config import setup_page

setup_page()
st.title("📊 CSV Visualizer & Forecast (Multipage)")
st.markdown("""
Welcome! Use the sidebar to navigate the workflow:  
- Upload your CSV  
- Confirm columns  
- Visualize, Forecast, or Chat with your CSV.
""")