import streamlit as st
import pandas as pd
import altair as alt

st.set_page_config(page_title="Pro Dashboard", layout="wide")

# Initialize session_state
if "csv_preview" not in st.session_state:
    st.session_state.csv_preview = None
if "xlsx_preview" not in st.session_state:
    st.session_state.xlsx_preview = None
if "page" not in st.session_state:
    st.session_state.page = "home"

# Navigation function
def navigate(page_name):
    st.session_state.page = page_name

# ---------------- Home Page ----------------
if st.session_state.page == "app":
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
            # Single-click navigation button
            if st.button(f"Go to {title}", key=f"home_{target_page}"):
                navigate(target_page)

    render_card(col1, "📄 CSV Page", st.session_state.csv_preview, "csv")
    render_card(col2, "📊 XLSX Page", st.session_state.xlsx_preview, "xlsx")