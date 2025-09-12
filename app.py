import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.io as pio
from prophet import Prophet
import seaborn as sns
import matplotlib.pyplot as plt
import io

st.set_page_config(page_title="CSV Visualizer with Forecasting (Interactive)", layout="wide")
st.title("📊 CSV Visualizer with Forecasting (Interactive)")

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
forecast_opacity = st.sidebar.slider("Forecast highlight opacity", 0.05, 1.0, 0.12, step=0.01)
show_confidence = st.sidebar.checkbox("Show confidence interval (upper/lower bounds)", True)

uploaded_file = st.file_uploader("Upload your CSV file (joined table)", type=["csv"])
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
    state_key = f"expand_{table_name.replace(' ', '_')}"
    if state_key not in st.session_state:
        st.session_state[state_key] = False
    btn_label = f"Minimise {table_name} Table" if st.session_state[state_key] else f"Expand {table_name} Table"
    clicked = st.button(btn_label, key=f"btn_{table_name}")
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
                file_name=f"{table_name.lower().replace(' ', '_')}.csv",
                mime="text/csv",
            )
            st.download_button(
                f"⬇️ Download {table_name} (Excel)",
                data=convert_df_to_excel(table_df),
                file_name=f"{table_name.lower().replace(' ', '_')}.xlsx",
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

date_col_sel = find_col_ci(selected_df, "date") or find_col_ci(selected_df, "Date")
amount_col_sel = find_col_ci(selected_df, "amount") or find_col_ci(selected_df, "Amount")
name_col_sel = find_col_ci(selected_df, "name") or find_col_ci(selected_df, "Name")

if date_col_sel:
    selected_df[date_col_sel] = pd.to_datetime(selected_df[date_col_sel], errors="coerce")
    selected_df = selected_df.sort_values(by=date_col_sel).reset_index(drop=True)
    selected_df['Year_Month'] = selected_df[date_col_sel].dt.to_period('M')
    selected_df['Year'] = selected_df[date_col_sel].dt.to_period('Y')
numerical_cols = selected_df.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = [c for c in selected_df.columns if c not in numerical_cols + ['Year_Month', 'Year', date_col_sel]]

st.subheader("📌 Column Selection for Visualization")
all_columns = selected_df.columns.tolist()
selected_columns = st.multiselect(
    "Select columns to include in visualization (include 'Date' and 'Amount' for forecasting)",
    all_columns,
    default=all_columns
)
if not selected_columns:
    st.warning("⚠️ Please select at least one column for visualization.")
    st.stop()
df_vis = selected_df[selected_columns].copy()
categorical_cols = df_vis.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
numerical_cols = df_vis.select_dtypes(include=[np.number]).columns.tolist()

st.subheader("📈 Interactive Visualization")
chart_options = [
    "Scatter Plot", "Line Chart", "Bar Chart", "Histogram", "Correlation Heatmap",
    "Seaborn Scatterplot", "Seaborn Boxplot", "Seaborn Violinplot", "Seaborn Pairplot",
    "Seaborn Heatmap", "Plotly Heatmap", "Treemap", "Sunburst", "Time-Series Decomposition"
]
chart_x_y_hue_req = {
    "Scatter Plot": (True, True, True),
    "Line Chart": (True, True, True),
    "Bar Chart": (True, True, True),
    "Histogram": (True, False, True),
    "Correlation Heatmap": (False, False, False),
    "Seaborn Scatterplot": (True, True, True),
    "Seaborn Boxplot": (True, True, True),
    "Seaborn Violinplot": (True, True, True),
    "Seaborn Pairplot": (False, False, True),
    "Seaborn Heatmap": (False, False, False),
    "Plotly Heatmap": (True, True, False),
    "Treemap": (True, True, False),
    "Sunburst": (True, True, False),
    "Time-Series Decomposition": (True, True, False)
}
chart_type = st.selectbox("Select Chart Type", chart_options)
need_x, need_y, need_hue = chart_x_y_hue_req.get(chart_type, (True, True, False))
x_col = st.selectbox("Select X Axis", df_vis.columns) if need_x else None
y_col = st.selectbox("Select Y Axis", [c for c in df_vis.columns if c != x_col]) if need_y else None
hue_col = None
if need_hue:
    hue_options = [c for c in df_vis.columns if c not in [x_col, y_col]]
    hue_col = st.selectbox("Select Hue/Category (optional)", ["(None)"] + hue_options)
    if hue_col == "(None)": hue_col = None

st.write("### Chart:")
try:
    if chart_type == "Scatter Plot":
        fig = px.scatter(df_vis, x=x_col, y=y_col, color=hue_col)
        st.plotly_chart(fig, use_container_width=True)
    elif chart_type == "Line Chart":
        fig = px.line(df_vis, x=x_col, y=y_col, color=hue_col)
        st.plotly_chart(fig, use_container_width=True)
    elif chart_type == "Bar Chart":
        fig = px.bar(df_vis, x=x_col, y=y_col, color=hue_col)
        st.plotly_chart(fig, use_container_width=True)
    elif chart_type == "Histogram":
        fig = px.histogram(df_vis, x=x_col, color=hue_col)
        st.plotly_chart(fig, use_container_width=True)
    elif chart_type == "Correlation Heatmap":
        if len(numerical_cols) >= 2:
            corr = df_vis[numerical_cols].corr()
            fig = px.imshow(corr, text_auto=True, color_continuous_scale='RdBu', aspect='auto')
            st.plotly_chart(fig, use_container_width=True)
except Exception as e:
    st.error(f"⚠️ Failed to render chart: {e}")

st.subheader("🔮 Forecasting (Optional)")
date_columns = [c for c in df_vis.columns if "date" in c.lower() or c.lower() in ["year_month", "year"]]
if date_columns and numerical_cols:
    selected_date_col = st.selectbox("Select Date Column for Forecasting", date_columns)
    selected_amount_col = st.selectbox("Select Numerical Column for Forecasting", numerical_cols)
    if selected_date_col and selected_amount_col:
        forecast_df = df_vis[[selected_date_col, selected_amount_col]].copy()
        forecast_df[selected_date_col] = pd.to_datetime(forecast_df[selected_date_col], errors="coerce")
        forecast_df[selected_amount_col] = pd.to_numeric(forecast_df[selected_amount_col], errors="coerce")
        forecast_df = forecast_df.dropna(subset=[selected_date_col, selected_amount_col])
        if len(forecast_df) >= 3:
            horizon = st.slider("Forecast Horizon (Months)", 3, 24, 6)
            forecast_df = forecast_df.rename(columns={selected_date_col:"ds", selected_amount_col:"y"})
            model = Prophet()
            model.fit(forecast_df)
            future = model.make_future_dataframe(periods=horizon, freq="M")
            forecast = model.predict(future)
            last_date = forecast_df["ds"].max()
            hist_forecast = forecast[forecast["ds"] <= last_date]
            future_forecast = forecast[forecast["ds"] > last_date]
            fig_forecast = px.line(hist_forecast, x="ds", y="yhat", title="Forecast")
            fig_forecast.add_scatter(x=future_forecast["ds"], y=future_forecast["yhat"], mode="lines", name="Forecast")

            if show_confidence:
                fig_forecast.add_scatter(x=forecast["ds"], y=forecast["yhat_upper"], mode="lines", name="Upper Bound", line=dict(dash="dot", color="green"))
                fig_forecast.add_scatter(x=forecast["ds"], y=forecast["yhat_lower"], mode="lines", name="Lower Bound", line=dict(dash="dot", color="red"))

            fig_forecast.add_vrect(x0=last_date, x1=forecast["ds"].max(), fillcolor=forecast_color, opacity=forecast_opacity, line_width=0, annotation_text="Forecast Period", annotation_position="top left")
            st.plotly_chart(fig_forecast, use_container_width=True)

            png_bytes_forecast = export_plotly_fig(fig_forecast)
            if png_bytes_forecast:
                st.download_button("⬇️ Download Forecast Chart (PNG)", data=png_bytes_forecast, file_name="forecast_chart.png", mime="image/png")

            forecast_table = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(horizon).copy()
            forecast_table.columns = ["Date", "Predicted", "Lower Bound", "Upper Bound"]
            forecast_table["Date"] = forecast_table["Date"].dt.strftime('%Y-%m-%d')
            forecast_table[["Predicted", "Lower Bound", "Upper Bound"]] = forecast_table[["Predicted", "Lower Bound", "Upper Bound"]].round(2)

            st.subheader("📅 Forecast Table (Future Predictions)")
            st.dataframe(forecast_table, use_container_width=True)

            col1, col2 = st.columns(2)
            with col1:
                st.download_button("⬇️ Download Forecast Data (CSV)", data=convert_df_to_csv(forecast_table), file_name="forecast_predictions.csv", mime="text/csv")
            with col2:
                st.download_button("⬇️ Download Forecast Data (Excel)", data=convert_df_to_excel(forecast_table), file_name="forecast_predictions.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

            with st.expander("📊 Forecast Summary Statistics"):
                future_avg = future_forecast["yhat"].mean()
                historical_avg = hist_forecast["yhat"].mean()
                growth_rate = ((future_avg - historical_avg) / historical_avg) * 100 if historical_avg != 0 else 0
                col1, col2, col3 = st.columns(3)
                with col1: st.metric("Historical Average", f"{historical_avg:,.2f}")
                with col2: st.metric("Forecast Average", f"{future_avg:,.2f}")
                with col3: st.metric("Growth Rate", f"{growth_rate:.2f}%")
        else:
            st.warning("⚠️ Need at least 3 data points for forecasting.")
else:
    st.info("ℹ️ To enable forecasting, include a valid date column and at least one numerical column in your selection.")