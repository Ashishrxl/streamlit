import streamlit as st
import pandas as pd

st.title("📊 XLSX Page")

uploaded_file = st.file_uploader("Upload XLSX", type=["xlsx"])
if uploaded_file:
    df = pd.read_excel(uploaded_file)
    st.dataframe(df)
    st.session_state.xlsx_preview = df  