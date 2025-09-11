import streamlit as st
import pandas as pd

st.title("📄 CSV Page")

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.dataframe(df)
    st.session_state.csv_preview = df