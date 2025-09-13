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

date_col_sel = find_col_ci(selected_df, "date") or find_col_ci(selected_df, "Date")
amount_col_sel = find_col_ci(selected_df, "amount") or find_col_ci(selected_df, "Amount")
name_col_sel = find_col_ci(selected_df, "name") or find_col_ci(selected_df, "Name")

st.subheader("📋 Selected & Processed Table")
state_key_processed = "expand_processed_table"
if state_key_processed not in st.session_state:
    st.session_state[state_key_processed] = False
btn_label_processed = f"Minimise Processed Table" if st.session_state[state_key_processed] else f"Expand Processed Table"
clicked_processed = st.button(btn_label_processed, key="btn_processed_table")
if clicked_processed:
    st.session_state[state_key_processed] = not st.session_state[state_key_processed]
if st.session_state[state_key_processed]:
    st.write(f"### {selected_table_name} - Processed Table (First 20 Rows)")
    st.dataframe(selected_df.head(20))
    with st.expander(f"📖 Show full Processed {selected_table_name} Table"):
        st.dataframe(selected_df)
    st.download_button(
        f"⬇️ Download Processed {selected_table_name} (CSV)",
        data=convert_df_to_csv(selected_df),
        file_name=f"processed_{selected_table_name.lower().replace(' ', '')}.csv",
        mime="text/csv",
    )
    st.download_button(
        f"⬇️ Download Processed {selected_table_name} (Excel)",
        data=convert_df_to_excel(selected_df),
        file_name=f"processed_{selected_table_name.lower().replace(' ', '')}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

st.subheader("📌 Column Selection for Visualization")
all_columns = selected_df.columns.tolist()
default_cols = all_columns.copy() if all_columns else []
selected_columns = st.multiselect(
    "Select columns to include in visualization (include 'Date' and 'Amount' for forecasting)",
    all_columns,
    default=default_cols,
    help="Choose which columns to include in your analysis and visualization"
)

if not selected_columns:
    st.warning("⚠️ Please select at least one column for visualization.")
    st.stop()

df_vis = selected_df[selected_columns].copy()
categorical_cols = df_vis.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
numerical_cols = df_vis.select_dtypes(include=[np.number]).columns.tolist()

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Total Columns", len(selected_columns))
with col2:
    st.metric("Numerical Columns", len(numerical_cols))
with col3:
    st.metric("Categorical Columns", len(categorical_cols))

with st.expander("📋 Column Details"):
    st.write("Categorical columns:", categorical_cols if categorical_cols else "None")
    st.write("Numerical columns:", numerical_cols if numerical_cols else "None")

st.subheader("🔮 Forecasting (Party + Bill Table)")
if not party_bill_df.empty:
    date_col_pb = find_col_ci(party_bill_df, "Date")
    amount_col_pb = find_col_ci(party_bill_df, "Amount")
    if date_col_pb and amount_col_pb:
        forecast_df = party_bill_df[[date_col_pb, amount_col_pb]].copy()
        forecast_df[date_col_pb] = pd.to_datetime(forecast_df[date_col_pb], errors="coerce")
        forecast_df[amount_col_pb] = pd.to_numeric(forecast_df[amount_col_pb], errors="coerce")
        forecast_df = forecast_df.dropna(subset=[date_col_pb, amount_col_pb])
        aggregation_period = st.selectbox("Select Aggregation Period", ["Monthly", "Yearly"])
        if aggregation_period == "Monthly":
            forecast_df = forecast_df.groupby(pd.Grouper(key=date_col_pb, freq='M')).sum(numeric_only=True).reset_index()
            freq_str = "M"
            period_type = "months"
        else:
            forecast_df = forecast_df.groupby(pd.Grouper(key=date_col_pb, freq='Y')).sum(numeric_only=True).reset_index()
            freq_str = "Y"
            period_type = "years"
        forecast_df = forecast_df.rename(columns={date_col_pb: "ds", amount_col_pb: "y"})
        min_data_points = 3
        if len(forecast_df) >= min_data_points:
            st.write(f"📈 **Forecasting based on {len(forecast_df)} data points**")
            col1, col2 = st.columns(2)
            with col1:
                if freq_str == "Y":
                    horizon = st.slider(f"Forecast Horizon ({period_type})", 1, 10, 3)
                else:
                    horizon = st.slider(f"Forecast Horizon ({period_type})", 3, 24, 6)
            with col2:
                st.write(f"**Data range:** {forecast_df['ds'].min().strftime('%Y-%m-%d')} to {forecast_df['ds'].max().strftime('%Y-%m-%d')}")
            with st.spinner("🔄 Running forecast model..."):
                model = Prophet()
                model.fit(forecast_df)
                future = model.make_future_dataframe(periods=horizon, freq=freq_str)
                forecast = model.predict(future)
            last_date = forecast_df["ds"].max()
            hist_forecast = forecast[forecast["ds"] <= last_date]
            future_forecast = forecast[forecast["ds"] > last_date]
            fig_forecast = px.line(
                hist_forecast, x="ds", y="yhat",
                labels={"ds": "Date", "yhat": "Amount"},
                title=f"Forecast Analysis - Next {horizon} {period_type.title()}"
            )
            fig_forecast.update_traces(selector=dict(mode="lines"), line=dict(color="blue", dash="solid"))
            fig_forecast.add_scatter(
                x=future_forecast["ds"], y=future_forecast["yhat"],
                mode="lines", name="Forecast", line=dict(color="orange", dash="dash")
            )
            if show_confidence:
                fig_forecast.add_scatter(
                    x=forecast["ds"], y=forecast["yhat_upper"],
                    mode="lines", name="Upper Bound", line=dict(dash="dot", color="green")
                )
                fig_forecast.add_scatter(
                    x=forecast["ds"], y=forecast["yhat_lower"],
                    mode="lines", name="Lower Bound", line=dict(dash="dot", color="red")
                )
            fig_forecast.add_vrect(
                x0=last_date, x1=forecast["ds"].max(),
                fillcolor=forecast_color, opacity=forecast_opacity, line_width=0,
                annotation_text="Forecast Period", annotation_position="top left"
            )
            st.plotly_chart(fig_forecast, use_container_width=True)
            png_bytes_forecast = export_plotly_fig(fig_forecast)
            if png_bytes_forecast:
                st.download_button(
                    "⬇️ Download Forecast Chart (PNG)",
                    data=png_bytes_forecast,
                    file_name="forecast_chart.png",
                    mime="image/png"
                )
            forecast_table = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(horizon).copy()
            forecast_table.columns = ["Date", "Predicted", "Lower Bound", "Upper Bound"]
            if freq_str == "Y":
                forecast_table["Date"] = forecast_table["Date"].dt.strftime('%Y')
            else:
                forecast_table["Date"] = forecast_table["Date"].dt.strftime('%Y-%m-%d')
            forecast_table["Predicted"] = forecast_table["Predicted"].round(2)
            forecast_table["Lower Bound"] = forecast_table["Lower Bound"].round(2)
            forecast_table["Upper Bound"] = forecast_table["Upper Bound"].round(2)
            st.subheader("📅 Forecast Table (Future Predictions)")
            st.dataframe(forecast_table, use_container_width=True)
            col1, col2 = st.columns(2)
            with col1:
                st.download_button(
                    "⬇️ Download Forecast Data (CSV)",
                    data=convert_df_to_csv(forecast_table),
                    file_name="forecast_predictions.csv",
                    mime="text/csv"
                )
            with col2:
                st.download_button(
                    "⬇️ Download Forecast Data (Excel)",
                    data=convert_df_to_excel(forecast_table),
                    file_name="forecast_predictions.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
            with st.expander("📊 Forecast Summary Statistics"):
                future_avg = future_forecast["yhat"].mean()
                historical_avg = hist_forecast["yhat"].mean()
                growth_rate = ((future_avg - historical_avg) / historical_avg) * 100 if historical_avg != 0 else 0
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Historical Average", f"{historical_avg:,.2f}")
                with col2:
                    st.metric("Forecast Average", f"{future_avg:,.2f}")
                with col3:
                    st.metric("Growth Rate", f"{growth_rate:.2f}%")
        else:
            st.warning(f"⚠️ Need at least 3 data points for forecasting.")
else:
    st.info("ℹ️ Party + Bill table with valid date and amount is required for forecasting.")

