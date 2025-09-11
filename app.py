import streamlit as st
import pandas as pd
import altair as alt

st.set_page_config(page_title="Pro Dashboard", layout="wide")
st.title("📊 Pro Analytics Dashboard")
st.write("Click a card to navigate, or see live previews of uploaded files.")

# Initialize session_state for previews
if "csv_preview" not in st.session_state:
    st.session_state.csv_preview = None
if "xlsx_preview" not in st.session_state:
    st.session_state.xlsx_preview = None

# CSS for card layout
card_style = """
<style>
.card {
    background-color: #f5f5f5;
    border-radius: 15px;
    padding: 15px;
    transition: 0.3s;
    box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    cursor: pointer;
}
.card:hover {
    background-color: #e0f7fa;
    transform: translateY(-3px);
}
.card-title {
    font-size: 24px;
    font-weight: bold;
    text-align: center;
}
.card-desc {
    font-size: 16px;
    color: #555;
    text-align: center;
    margin-bottom: 10px;
}
.card-preview {
    margin-top: 10px;
    max-height: 80px;
    overflow: auto;
    font-size: 12px;
}
</style>
"""
st.markdown(card_style, unsafe_allow_html=True)

# Layout columns
col1, col2 = st.columns(2)

def render_clickable_card(title, desc, df, target_page):
    with st.form(key=title):
        st.markdown(f'<div class="card">', unsafe_allow_html=True)
        st.markdown(f'<div class="card-title">{title}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="card-desc">{desc}</div>', unsafe_allow_html=True)
        # Show mini table preview
        if df is not None:
            st.markdown(f'<div class="card-preview">{df.head(3).to_html(index=False)}</div>', unsafe_allow_html=True)
            numeric_cols = df.select_dtypes(include='number').columns
            for col in numeric_cols[:2]:
                chart = alt.Chart(df.head(10)).mark_line(point=True).encode(
                    x=alt.X(df.head(10).index, title='Index'),
                    y=alt.Y(col, title=col)
                )
                st.altair_chart(chart, use_container_width=True)
        # Submit button covers the whole card
        submitted = st.form_submit_button(label="Click anywhere on the card")
        st.markdown("</div>", unsafe_allow_html=True)
        if submitted:
            st.experimental_set_query_params(page=target_page)
            st.experimental_rerun()

# Render cards
with col1:
    render_clickable_card("📄 CSV Page", "Upload CSV files and view analytics.", st.session_state.csv_preview, "csv")

with col2:
    render_clickable_card("📊 XLSX Page", "Upload XLSX files and view analytics.", st.session_state.xlsx_preview, "xlsx")