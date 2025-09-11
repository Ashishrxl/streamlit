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
    .nav-bar {
        display: flex;
        justify-content: center;
        gap: 2rem;
        margin-bottom: 2rem;
    }
    .nav-bar button {
        font-size: 16px;
        padding: 0.5rem 1rem;
        border-radius: 8px;
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

    if st.session_state.csv_preview is not None:
        st.subheader("📄 CSV Preview")
        st.dataframe(st.session_state.csv_preview.head(5))
        numeric_cols = st.session_state.csv_preview.select_dtypes(include="number").columns
        if len(numeric_cols) > 0:
            chart = alt.Chart(st.session_state.csv_preview.head(10)).mark_line(point=True).encode(
                x=alt.X(st.session_state.csv_preview.head(10).index, title="Index"),
                y=alt.Y(numeric_cols[0], title=numeric_cols[0]),
            )
            st.altair_chart(chart, use_container_width=True)

    if st.session_state.xlsx_preview is not None:
        st.subheader("📊 XLSX Preview")
        st.dataframe(st.session_state.xlsx_preview.head(5))

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