import streamlit as st
import pandas as pd
import altair as alt

st.set_page_config(page_title="Pro Dashboard", layout="wide")

# Initialize session_state for previews
if "csv_preview" not in st.session_state:
    st.session_state.csv_preview = None
if "xlsx_preview" not in st.session_state:
    st.session_state.xlsx_preview = None

# Determine current page from query params
page = st.query_params.get("page", ["home"])[0]

# ---------------- Home Page ----------------
if page == "home":
    st.title("📊 Pro Dashboard Home")
    st.write("Click a button to navigate or see live previews.")

    col1, col2 = st.columns(2)

    def render_card(col, title, df, target_page):
        with col:
            st.markdown(f"### {title}")
            if df is not None:
                st.dataframe(df.head(3))
                numeric_cols = df.select_dtypes(include='number').columns
                for col_name in numeric_cols[:2]:
                    chart = alt.Chart(df.head(10)).mark_line(point=True).encode(
                        x=alt.X(df.head(10).index, title='Index'),
                        y=alt.Y(col_name, title=col_name)
                    )
                    st.altair_chart(chart, use_container_width=True)
            if st.button(f"Go to {title}", key=title):
                st.query_params = {"page": target_page}

    render_card(col1, "📄 CSV Page", st.session_state.csv_preview, "csv")
    render_card(col2, "📊 XLSX Page", st.session_state.xlsx_preview, "xlsx")

# ---------------- CSV Page ----------------
elif page == "csv":
    st.title("📄 CSV Page")
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.dataframe(df)
        st.session_state.csv_preview = df

    st.write("---")
    if st.button("🏠 Home Page"):
        st.query_params = {"page": "home"}
    if st.button("📊 XLSX Page"):
        st.query_params = {"page": "xlsx"}

# ---------------- XLSX Page ----------------
elif page == "xlsx":
    st.title("📊 XLSX Page")
    uploaded_file = st.file_uploader("Upload XLSX", type=["xlsx"])
    if uploaded_file:
        df = pd.read_excel(uploaded_file)
        st.dataframe(df)
        st.session_state.xlsx_preview = df

    st.write("---")
    if st.button("🏠 Home Page"):
        st.query_params = {"page": "home"}
    if st.button("📄 CSV Page"):
        st.query_params = {"page": "csv"}