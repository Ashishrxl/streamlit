import streamlit as st
from core import run_app_logic

st.header("Step 2: Visualization")

if 'uploaded_df' in st.session_state:
    run_app_logic(
        st.session_state['uploaded_df'],
        False,      # or True if it's 'alldata.csv'
        None,       # You can pass llm if set in session_state
        "#FFA500",  # color
        0.12,       # opacity
        True        # show_confidence
    )
else:
    st.warning("Please upload your data and confirm columns first (Step 1).")