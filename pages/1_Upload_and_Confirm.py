import streamlit as st
import pandas as pd
from rename_section import handle_renaming_flow

st.header("Step 1: Upload and Confirm")
uploaded_file = st.file_uploader("Upload your CSV file!", type=["csv"])

if uploaded_file is not None:
    st.session_state['uploaded_file'] = uploaded_file
    # This should use a function like rename_section.handle_renaming_flow, but adapted for multipages.
    # E.g., save df in st.session_state['uploaded_df']
    # For demo:
    try:
        df = pd.read_csv(uploaded_file, low_memory=False)
        st.session_state['uploaded_df'] = df
        st.success("✅ File uploaded and loaded successfully! Go to the next step in the sidebar.")
    except Exception as e:
        st.error(f"❌ Error reading CSV: {e}")
else:
    st.warning("Upload your file to proceed.")