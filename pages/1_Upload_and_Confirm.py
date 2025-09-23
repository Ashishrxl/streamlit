import streamlit as st
import pandas as pd
from rename_section import handle_renaming_flow

st.header("Step 1: Upload and Confirm")

uploaded_file = st.file_uploader("Upload your CSV file!", type=["csv"])

if uploaded_file is not None:
    st.session_state['uploaded_file'] = uploaded_file  # Save file object for other pages

    if uploaded_file.name.lower() == "alldata.csv":
        # Directly load alldata.csv without rename prompt
        try:
            df = pd.read_csv(uploaded_file, low_memory=False)
            st.session_state['uploaded_df'] = df
            st.success("✅ alldata.csv uploaded and loaded successfully!")
            st.info("Navigate to Visualization or Chat pages.")
        except Exception as e:
            st.error(f"❌ Error reading alldata.csv: {e}")
    else:
        # Trigger rename confirmation for all other files
        handle_renaming_flow(
            uploaded_file,
            None,      # run_app_logic to be called in visualization page
            None,      # llm will be set in app main or session state
            None,
            None,
            None
        )
else:
    st.warning("Please upload your CSV file to proceed.")