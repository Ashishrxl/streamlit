import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.io as pio
from prophet import Prophet
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
import io

st.set_page_config(page_title="CSV Visualizer with Forecasting (Interactive)", layout="wide")
st.title("📊 CSV Visualizer with Forecasting (Interactive)")

if 'chart_rendered' not in st.session_state:
    st.session_state.chart_rendered = False
if 'chart_type_selected' not in st.session_state:
    st.session_state.chart_type_selected = "Scatter Plot"
if 'x_col_selected' not in st.session_state:
    st.session_state.x_col_selected = None
if 'y_col_selected' not in st.session_state:
    st.session_state.y_col_selected = None
if 'hue_col_selected' not in st.session_state:
    st.session_state.hue_col_selected = None
if 'forecast_section_stage' not in st.session_state:
    st.session_state.forecast_section_stage = "select_columns"
if 'forecast_selected_date_col' not in st.session_state:
    st.session_state.forecast_selected_date_col = None
if 'forecast_selected_amount_col' not in st.session_state:
    st.session_state.forecast_selected_amount_col = None
if 'forecast_aggregation_period' not in st.session_state:
    st.session_state.forecast_aggregation_period = None
if 'forecast_submitted' not in st.session_state:
    st.session_state.forecast_submitted = False

hide_streamlit_style = """
<style>                      
#MainMenu, footer, header {visibility: hidden;}                      
footer {display: none !important;}                      
header {display: none !important;}                      
#MainMenu {display: none !important;}                      
[data-testid="stToolbar"] { display: none !important; }                      
.st-emotion-cache-1xw8zd0 {display: none !important;}                      
[aria-label="View app source"] {display: none !important;}                      
a[href^="https://github.com"] {display: none !important;}                      
[data-testid="stDecoration"] {display: none !important;}                      
[data-testid="stStatusWidget"] {display: none !important;}                      
button[title="Menu"] {display: none !important;}
.element-container:has(> .stSelectbox) {
    position: relative !important;
}
</style>                
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

def find_col_ci(df: pd.DataFrame, target: str):
    for c in df.columns:
        if c.lower() == target.lower():
            return c
    return None

def convert_df_to_csv(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")

def convert_df_to_excel(df: pd.DataFrame) -> bytes:
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
        df.to_excel(writer, index=False, sheet_name="Sheet1")
    return buffer.getvalue()

def export_plotly_fig(fig):
    try:
        return pio.to_image(fig, format="png", engine="kaleido")
    except Exception:
        return None

def export_matplotlib_fig(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    return buf.getvalue()

st.sidebar.header("⚙️ Settings")
forecast_color = st.sidebar.color_picker("Forecast highlight color", "#FFA500")
forecast_opacity = st.sidebar.slider("Forecast highlight opacity", 0.05, 1.0, 0.2, step=0.05)
show_confidence = st.sidebar.checkbox("Show confidence interval (upper/lower bounds)", True)

uploaded_file = st.file_uploader("Upload your CSV file !", type=["csv"])
if uploaded_file is None:
    st.info("Upload a CSV to start. The app will derive tables and let you visualize/forecast.")
    st.stop()

try:
    uploaded_df = pd.read_csv(uploaded_file, low_memory=False)
except Exception as e:
    st.error(f"❌ Error reading CSV: {e}")
    st.stop()

st.success("✅ File uploaded successfully!")

id_col = find_col_ci(uploaded_df, "ID")
name_col = find_col_ci(uploaded_df, "Name")
party_df = uploaded_df[[id_col, name_col]].drop_duplicates().reset_index(drop=True) if id_col and name_col else pd.DataFrame()

bill_col = find_col_ci(uploaded_df, "Bill")
partyid_col = find_col_ci(uploaded_df, "PartyId")
date_col_master = find_col_ci(uploaded_df, "Date")
amount_col_master = find_col_ci(uploaded_df, "Amount")
bill_df = (
    uploaded_df[[bill_col, partyid_col, date_col_master, amount_col_master]].drop_duplicates().reset_index(drop=True)
    if bill_col and partyid_col and date_col_master and amount_col_master else pd.DataFrame()
)

billdetails_cols = [find_col_ci(uploaded_df, c) for c in ["IndexId", "Billindex", "Item", "Qty", "Rate", "Less"]]
billdetails_cols = [c for c in billdetails_cols if c]
billdetails_df = uploaded_df[billdetails_cols].drop_duplicates().reset_index(drop=True) if billdetails_cols else pd.DataFrame()

try:
    party_bill_df = pd.merge(
        party_df, bill_df, left_on=id_col, right_on=partyid_col, how="inner", suffixes=("_party", "_bill")
    ) if not party_df.empty and not bill_df.empty else pd.DataFrame()
except Exception:
    party_bill_df = pd.DataFrame()

try:
    billindex_col = find_col_ci(uploaded_df, "Billindex")
    bill_billdetails_df = pd.merge(
        bill_df, billdetails_df, left_on=bill_col, right_on=billindex_col, how="inner", suffixes=("_bill", "_details")
    ) if not bill_df.empty and not billdetails_df.empty else pd.DataFrame()
except Exception:
    bill_billdetails_df = pd.DataFrame()

st.subheader("🗂️ Tables Preview")
tables_dict = {
    "Uploaded Table": uploaded_df,
    "Party": party_df,
    "Bill": bill_df,
    "BillDetails": billdetails_df,
    "Party + Bill": party_bill_df,
    "Bill + BillDetails": bill_billdetails_df
}

for table_name, table_df in tables_dict.items():
    state_key = f"expand_{table_name.replace(' ', '')}"
    if state_key not in st.session_state:
        st.session_state[state_key] = False
    btn_label = f"Minimise {table_name} Table" if st.session_state[state_key] else f"Expand {table_name} Table"
    clicked = st.button(btn_label, key=f"btn{table_name}")
    if clicked:
        st.session_state[state_key] = not st.session_state[state_key]
    if st.session_state[state_key]:
        st.write(f"### {table_name} Table (First 20 Rows)")
        if not table_df.empty:
            st.dataframe(table_df.head(20))
            with st.expander(f"📖 Show full {table_name} Table"):
                st.dataframe(table_df)
            st.download_button(
                f"⬇️ Download {table_name} (CSV)",
                data=convert_df_to_csv(table_df),
                file_name=f"{table_name.lower().replace(' ', '')}.csv",
                mime="text/csv",
            )
            st.download_button(
                f"⬇️ Download {table_name} (Excel)",
                data=convert_df_to_excel(table_df),
                file_name=f"{table_name.lower().replace(' ', '')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
        else:
            st.info("ℹ️ Not available from the uploaded CSV.")

st.subheader("📌 Select Table for Visualization")
available_tables = {k: v for k, v in tables_dict.items() if not v.empty}
if not available_tables:
    st.warning("⚠️ No usable tables could be derived from the uploaded CSV.")
    st.stop()

selected_table_name = st.selectbox("Select one table", list(available_tables.keys()))
selected_df = available_tables[selected_table_name].copy()

if 'selected_table_prev' not in st.session_state:
    st.session_state['selected_table_prev'] = selected_table_name
if st.session_state['selected_table_prev'] != selected_table_name:
    st.session_state.chart_rendered = False
    st.session_state.chart_type_selected = "Scatter Plot"
    st.session_state.x_col_selected = None
    st.session_state.y_col_selected = None
    st.session_state.hue_col_selected = None
    st.session_state['selected_table_prev'] = selected_table_name

date_col_sel = find_col_ci(selected_df, "date") or find_col_ci(selected_df, "Date")
amount_col_sel = find_col_ci(selected_df, "amount") or find_col_ci(selected_df, "Amount")
name_col_sel = find_col_ci(selected_df, "name") or find_col_ci(selected_df, "Name")

if date_col_sel and amount_col_sel:
    try:
        selected_df[date_col_sel] = pd.to_datetime(selected_df[date_col_sel], errors="coerce")
        selected_df = selected_df.dropna(subset=[date_col_sel, amount_col_sel])
        selected_df[amount_col_sel] = pd.to_numeric(selected_df[amount_col_sel], errors="coerce")
        selected_df = selected_df.dropna(subset=[amount_col_sel])
        selected_df["Year_Month"] = selected_df[date_col_sel].dt.to_period("M").astype(str)
        monthly_grouped = selected_df.groupby("Year_Month")[amount_col_sel].sum().reset_index()
    except Exception:
        monthly_grouped = pd.DataFrame()
else:
    monthly_grouped = pd.DataFrame()

st.subheader("📈 Visualization")
with st.form("chart_form"):
    chart_type = st.selectbox(
        "Select Chart Type",
        ["Scatter Plot", "Bar Chart", "Line Chart", "Histogram", "Box Plot"],
        index=["Scatter Plot", "Bar Chart", "Line Chart", "Histogram", "Box Plot"].index(st.session_state.chart_type_selected),
    )
    x_col = st.selectbox("X-axis Column", selected_df.columns, index=0 if st.session_state.x_col_selected is None else selected_df.columns.get_loc(st.session_state.x_col_selected))
    y_col = st.selectbox("Y-axis Column", selected_df.columns, index=0 if st.session_state.y_col_selected is None else selected_df.columns.get_loc(st.session_state.y_col_selected))
    hue_col = st.selectbox("Hue Column (Optional)", ["None"] + list(selected_df.columns), index=0 if st.session_state.hue_col_selected is None else (selected_df.columns.get_loc(st.session_state.hue_col_selected) + 1))
    submitted = st.form_submit_button("Generate Chart")
    if submitted:
        st.session_state.chart_type_selected = chart_type
        st.session_state.x_col_selected = x_col
        st.session_state.y_col_selected = y_col
        st.session_state.hue_col_selected = None if hue_col == "None" else hue_col
        st.session_state.chart_rendered = True

if st.session_state.chart_rendered:
    chart_type = st.session_state.chart_type_selected
    x_col = st.session_state.x_col_selected
    y_col = st.session_state.y_col_selected
    hue_col = st.session_state.hue_col_selected
    fig = None
    if chart_type == "Scatter Plot":
        fig = px.scatter(selected_df, x=x_col, y=y_col, color=hue_col if hue_col else None)
    elif chart_type == "Bar Chart":
        fig = px.bar(selected_df, x=x_col, y=y_col, color=hue_col if hue_col else None)
    elif chart_type == "Line Chart":
        fig = px.line(selected_df, x=x_col, y=y_col, color=hue_col if hue_col else None)
    elif chart_type == "Histogram":
        fig = px.histogram(selected_df, x=x_col, color=hue_col if hue_col else None)
    elif chart_type == "Box Plot":
        fig = px.box(selected_df, x=x_col, y=y_col, color=hue_col if hue_col else None)
    if fig is not None:
        st.plotly_chart(fig, use_container_width=True)
        img_bytes = export_plotly_fig(fig)
        if img_bytes:
            st.download_button(
                "⬇️ Download Chart as PNG",
                data=img_bytes,
                file_name=f"{chart_type.lower().replace(' ', '_')}.png",
                mime="image/png",
            )

if not monthly_grouped.empty:
    st.subheader("📅 Forecasting with Prophet")
    with st.form("forecast_form"):
        forecast_periods = st.number_input("Forecast periods (months)", min_value=1, max_value=60, value=24)
        forecast_submitted = st.form_submit_button("Run Forecast")
        if forecast_submitted:
            st.session_state.forecast_submitted = True
            st.session_state.forecast_periods = forecast_periods

    if st.session_state.forecast_submitted:
        try:
            df_prophet = monthly_grouped.rename(columns={"Year_Month": "ds", amount_col_sel: "y"}).copy()
            df_prophet["ds"] = pd.to_datetime(df_prophet["ds"])
            model = Prophet(interval_width=0.95)
            model.fit(df_prophet)
            future = model.make_future_dataframe(periods=st.session_state.forecast_periods, freq="M")
            forecast = model.predict(future)
            fig1 = px.line(df_prophet, x="ds", y="y", title="Historical Data")
            st.plotly_chart(fig1, use_container_width=True)

            fig2 = px.line(forecast, x="ds", y="yhat", title=f"Forecast - Next {st.session_state.forecast_periods} Months")
            forecast_start = df_prophet["ds"].max()
            forecast_end = forecast["ds"].max()
            fig2.add_vrect(x0=forecast_start, x1=forecast_end,
                           fillcolor=forecast_color, opacity=forecast_opacity,
                           layer="below", line_width=0)
            if show_confidence:
                fig2.add_traces([
                    dict(type="scatter", x=forecast["ds"], y=forecast["yhat_upper"],
                         mode="lines", name="Upper Bound",
                         line=dict(dash="dot", color=forecast_color),
                         opacity=forecast_opacity),
                    dict(type="scatter", x=forecast["ds"], y=forecast["yhat_lower"],
                         mode="lines", name="Lower Bound",
                         line=dict(dash="dot", color=forecast_color),
                         opacity=forecast_opacity)
                ])
            st.plotly_chart(fig2, use_container_width=True)

            st.download_button(
                "⬇️ Download Forecast Data (CSV)",
                data=convert_df_to_csv(forecast),
                file_name="forecast.csv",
                mime="text/csv",
            )
            st.download_button(
                "⬇️ Download Forecast Data (Excel)",
                data=convert_df_to_excel(forecast),
                file_name="forecast.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )

            forecast_img = export_plotly_fig(fig2)
            if forecast_img:
                st.download_button(
                    "⬇️ Download Forecast Chart as PNG",
                    data=forecast_img,
                    file_name="forecast_chart.png",
                    mime="image/png",
                )
        except Exception as e:
            st.error(f"❌ Forecasting failed: {e}")

