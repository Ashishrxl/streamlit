import streamlit as st
import pandas as pd
import altair as alt

st.set_page_config(page_title="Pro Dashboard", layout="wide")

# --- Session state ---
if "page" not in st.session_state:
    st.session_state.page = "home"
if "csv_preview" not in st.session_state:
    st.session_state.csv_preview = None
if "xlsx_preview" not in st.session_state:
    st.session_state.xlsx_preview = None

# --- Navigation bar ---
st.markdown(
    """
    <style>
    .nav-container {
        display: flex;
        justify-content: center;
        gap: 2rem;
        margin-bottom: 1.5rem;
    }
    button[kind="primary"] {
        background-color: #4CAF50 !important;
        color: white !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

col1, col2, col3 = st.columns(3)
with col1:
    if st.button("🏠 Home"):
        st.session_state.page = "home"
with col2:
    if st.button("📄 CSV"):
        st.session_state.page = "csv"
with col3:
    if st.button("📊 XLSX"):
        st.session_state.page = "xlsx"

st.write("---")

# --- Pages ---
if st.session_state.page == "home":
    st.title("📊 Pro Dashboard Home")
    st.write("Upload CSV/XLSX files on their pages and see previews here.")

    # CSV Preview
    if st.session_state.csv_preview is not None:
        st.subheader("📄 CSV Preview")
        st.dataframe(st.session_state.csv_preview.head(5))

        numeric_cols = st.session_state.csv_preview.select_dtypes(include="number").columns
        if len(numeric_cols) > 0:
            df_preview = (
                st.session_state.csv_preview.head(10)
                .reset_index()
                .rename(columns={"index": "Row"})
            )
            chart = (
                alt.Chart(df_preview)
                .mark_line(point=True)
                .encode(
                    x=alt.X("Row", title="Index"),
                    y=alt.Y(numeric_cols[0], title=numeric_cols[0]),
                )
            )
            st.altair_chart(chart, use_container_width=True)

    # XLSX Preview
    if st.session_state.xlsx_preview is not None:
        st.subheader("📊 XLSX Preview")
        st.dataframe(st.session_state.xlsx_preview.head(5))

        numeric_cols = st.session_state.xlsx_preview.select_dtypes(include="number").columns
        if len(numeric_cols) > 0:
            df_preview = (
                st.session_state.xlsx_preview.head(10)
                .reset_index()
                .rename(columns={"index": "Row"})
            )
            chart = (
                alt.Chart(df_preview)
                .mark_line(point=True)
                .encode(
                    x=alt.X("Row", title="Index"),
                    y=alt.Y(numeric_cols[0], title=numeric_cols[0]),
                )
            )
            st.altair_chart(chart, use_container_width=True)

elif st.session_state.page == "csv":
    st.title("📄 CSV Page")
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.dataframe(df)
        st.session_state.csv_preview = df

elif st.session_state.page == "xlsx":
    st.title("📊 XLSX Page")
    uploaded_file = st.file_uploader("Upload XLSX", type=["xlsx"])
    if uploaded_file:
        df = pd.read_excel(uploaded_file)
        st.dataframe(df)
        st.session_state.xlsx_preview = df