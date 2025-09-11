import streamlit as st
import pandas as pd

st.set_page_config(page_title="XLSX Page", layout="wide")
st.title("📊 XLSX Page")

uploaded_file = st.file_uploader("Upload XLSX", type=["xlsx"])
if uploaded_file:
    df = pd.read_excel(uploaded_file)
    st.dataframe(df)
    st.session_state.xlsx_preview = df  # live preview for dashboard

st.write("---")
st.write("Navigate:")

col1, col2 = st.columns(2)
with col1:
    if st.button("📄 CSV Page"):
        st.experimental_set_query_params(page="csv")
        st.experimental_rerun()
with col2:
    if st.button("🏠 Home Page"):
        st.experimental_set_query_params(page="app")
        st.experimental_rerun()