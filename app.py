import streamlit as st
import pandas as pd
import altair as alt

st.set_page_config(page_title="Pro Dashboard", layout="wide")
st.title("📊 Pro Analytics Dashboard")
st.write("View live previews and sparkline charts for your uploaded CSV/XLSX files.")

# Initialize session_state for previews
if "csv_preview" not in st.session_state:
    st.session_state.csv_preview = None
if "xlsx_preview" not in st.session_state:
    st.session_state.xlsx_preview = None

# CSS for cards
card_style = """
<style>
.card {
    background-color: #f5f5f5;
    border-radius: 15px;
    padding: 20px;
    text-align: center;
    transition: 0.3s;
    cursor: pointer;
    box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    height: 300px;
}
.card:hover {
    background-color: #e0f7fa;
    transform: translateY(-5px);
}
.card-title {
    font-size: 24px;
    font-weight: bold;
}
.card-desc {
    font-size: 16px;
    color: #555;
}
.card-preview {
    margin-top: 10px;
    max-height: 80px;
    overflow: auto;
    font-size: 12px;
    text-align: left;
}
.card-chart {
    margin-top: 5px;
}
</style>
"""
st.markdown(card_style, unsafe_allow_html=True)

# Layout cards
col1, col2 = st.columns(2)

def render_card(title, desc, df):
    html = f"""
    <div class="card" onclick="window.location.href='./{title.lower().split()[0]}'">
        <div class="card-title">{title}</div>
        <div class="card-desc">{desc}</div>
    """
    # Show mini table preview
    if df is not None:
        html += f'<div class="card-preview">{df.head(3).to_html(index=False)}</div>'
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)
    # Render sparklines for all numeric columns
    if df is not None:
        numeric_cols = df.select_dtypes(include='number').columns
        for col in numeric_cols:
            sparkline = alt.Chart(df.head(10)).mark_line(point=True).encode(
                x=alt.X('index', title='Index'),
                y=alt.Y(col, title=col)
            ).transform_calculate(
                index='datum.index || 0'
            )
            st.altair_chart(sparkline, use_container_width=True)

# Render CSV Card
with col1:
    render_card(
        title="📄 CSV Page",
        desc="Upload CSV files and view analytics.",
        df=st.session_state.csv_preview
    )

# Render XLSX Card
with col2:
    render_card(
        title="📊 XLSX Page",
        desc="Upload XLSX files and view analytics.",
        df=st.session_state.xlsx_preview
    )