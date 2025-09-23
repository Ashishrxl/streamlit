import streamlit as st
from modules.chat_section import chat_with_csv_section

st.header("Step 4: Chat with your CSV")

if 'uploaded_df' in st.session_state:
    tables_dict = {"Uploaded Table": st.session_state['uploaded_df']}
    # (or build tables_dict if you use complex extracted tables)
    # Optionally setup llm here
    llm = None
    chat_with_csv_section(tables_dict, llm)
else:
    st.warning("Upload data and confirm columns first (Step 1).")