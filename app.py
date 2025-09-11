import streamlit as st
import altair as alt

st.set_page_config(page_title="Pro Dashboard", layout="wide")

st.title("📊 Pro Dashboard Home")
st.write("Navigate using the sidebar →")

# Show previews if available
if "csv_preview" in st.session_state and st.session_state.csv_preview is not None:
    st.markdown("### 📄 CSV Preview")
    st.dataframe(st.session_state.csv_preview.head(5))

    numeric_cols = st.session_state.csv_preview.select_dtypes(include='number').columns
    if len(numeric_cols) > 0:
        chart = alt.Chart(st.session_state.csv_preview.head(10)).mark_line(point=True).encode(
            x=alt.X(st.session_state.csv_preview.head(10).index, title="Index"),
            y=alt.Y(numeric_cols[0], title=numeric_cols[0])
        )
        st.altair_chart(chart, use_container_width=True)

if "xlsx_preview" in st.session_state and st.session_state.xlsx_preview is not None:
    st.markdown("### 📊 XLSX Preview")
    st.dataframe(st.session_state.xlsx_preview.head(5))