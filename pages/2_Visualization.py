import streamlit as st
from core import run_app_logic

st.header("Step 2: Visualize and Explore")

if 'uploaded_df' in st.session_state and 'uploaded_file' in st.session_state:
    # Check uploaded filename to see if it equals 'alldata.csv' (case insensitive)
    uploaded_filename = st.session_state['uploaded_file'].name if st.session_state['uploaded_file'] else ""
    is_alldata = (uploaded_filename.lower() == "alldata.csv")

    run_app_logic(
        st.session_state['uploaded_df'],
        is_alldata,
        st.session_state.get('llm', None),
        st.session_state.get('forecast_color', "#FFA500"),
        st.session_state.get('forecast_opacity', 0.12),
        st.session_state.get('show_confidence', True)
    )
else:
    st.warning("Please upload and confirm your data in Step 1 before visualization.")