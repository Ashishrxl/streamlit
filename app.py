import streamlit as st 
import pandas as pd 
import numpy as np 
import plotly.express as px 
import plotly.io as pio 
from prophet import Prophet 
import seaborn as sns 
import matplotlib.pyplot as plt 
from statsmodels.tsa.seasonal import seasonal_decompose 
import io as sys_io

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
</style>""" 

st.markdown(hide_streamlit_style, unsafe_allow_html=True)



def find_col_ci(df: pd.DataFrame, target: str): 
    for c in df.columns: 
        if c.lower() == target.lower():
        return c 
    return None

def convert_df_to_csv(df: pd.DataFrame) -> bytes: return df.to_csv(index=False).encode("utf-8")

def convert_df_to_excel(df: pd.DataFrame) -> bytes: buffer = sys_io.BytesIO() with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer: df.to_excel(writer, index=False, sheet_name="Sheet1") return buffer.getvalue()

def export_plotly_fig(fig): try: return pio.to_image(fig, format="png", engine="kaleido") except Exception: return None

def export_matplotlib_fig(fig): buf = sys_io.BytesIO() fig.savefig(buf, format="png", bbox_inches="tight") buf.seek(0) return buf.getvalue()



st.sidebar.header("⚙️ Settings") forecast_color = st.sidebar.color_picker("Forecast highlight color", "#FFA500") forecast_opacity = st.sidebar.slider("Forecast highlight opacity", 0.05, 1.0, 0.12, step=0.01) show_confidence = st.sidebar.checkbox("Show confidence interval (upper/lower bounds)", True)


uploaded_file = st.file_uploader("Upload your CSV file (joined table)", type=["csv"]) if uploaded_file is None: st.info("Upload a CSV to start. The app will derive tables and let you visualize/forecast.") st.stop()

try: uploaded_df = pd.read_csv(uploaded_file, low_memory=False) except Exception as e: st.error(f"❌ Error reading CSV: {e}") st.stop()

st.success("✅ File uploaded successfully!")



id_col = find_col_ci(uploaded_df, "ID") name_col = find_col_ci(uploaded_df, "Name") party_df = uploaded_df[[id_col, name_col]].drop_duplicates().reset_index(drop=True) if id_col and name_col else pd.DataFrame()

bill_col = find_col_ci(uploaded_df, "Bill") partyid_col = find_col_ci(uploaded_df, "PartyId") date_col_master = find_col_ci(uploaded_df, "Date") amount_col_master = find_col_ci(uploaded_df, "Amount") bill_df = ( uploaded_df[[bill_col, partyid_col, date_col_master, amount_col_master]].drop_duplicates().reset_index(drop=True) if bill_col and partyid_col and date_col_master and amount_col_master else pd.DataFrame() )

billdetails_cols = [find_col_ci(uploaded_df, c) for c in ["IndexId", "Billindex", "Item", "Qty", "Rate", "Less"]] billdetails_cols = [c for c in billdetails_cols if c] billdetails_df = uploaded_df[billdetails_cols].drop_duplicates().reset_index(drop=True) if billdetails_cols else pd.DataFrame()

try: party_bill_df = pd.merge( party_df, bill_df, left_on=id_col, right_on=partyid_col, how="inner", suffixes=("_party", "_bill") ) if not party_df.empty and not bill_df.empty else pd.DataFrame() except Exception: party_bill_df = pd.DataFrame()

try: billindex_col = find_col_ci(uploaded_df, "Billindex") bill_billdetails_df = pd.merge( bill_df, billdetails_df, left_on=bill_col, right_on=billindex_col, how="inner", suffixes=("_bill", "_details") ) if not bill_df.empty and not billdetails_df.empty else pd.DataFrame() except Exception: bill_billdetails_df = pd.DataFrame()



st.subheader("🗂️ Tables Preview") tables_dict = { "Uploaded Table": uploaded_df, "Party": party_df, "Bill": bill_df, "BillDetails": billdetails_df, "Party + Bill": party_bill_df, "Bill + BillDetails": bill_billdetails_df }

for table_name, table_df in tables_dict.items(): state_key = f"expand_{table_name.replace(' ', '_')}" if state_key not in st.session_state: st.session_state[state_key] = False

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



st.subheader("📌 Select Table for Visualization") available_tables = {k: v for k, v in tables_dict.items() if not v.empty} if not available_tables: st.warning("⚠️ No usable tables could be derived from the uploaded CSV.") st.stop()

selected_table_name = st.selectbox("Select one table", list(available_tables.keys())) selected_df = available_tables[selected_table_name].copy()



date_col_sel = find_col_ci(selected_df, "date") or find_col_ci(selected_df, "Date") amount_col_sel = find_col_ci(selected_df, "amount") or find_col_ci(selected_df, "Amount")

if date_col_sel: try: selected_df[date_col_sel] = pd.to_datetime(selected_df[date_col_sel], errors="coerce") selected_df = selected_df.sort_values(by=date_col_sel).reset_index(drop=True) selected_df['Year_Month'] = selected_df[date_col_sel].dt.to_period('M') selected_df['Year'] = selected_df[date_col_sel].dt.to_period('Y')

st.markdown("### 📅 Data Aggregation Options")
    time_period = st.selectbox(
        "Choose time aggregation period:",
        ["No Aggregation", "Monthly", "Yearly"]
    )

    if time_period != "No Aggregation":
        period_col = 'Year_Month' if time_period == "Monthly" else 'Year'
        freq_setting = "M" if time_period == "Monthly" else "Y"
        grouped_df = selected_df.groupby(period_col, as_index=False)[amount_col_sel].sum()
        grouped_df[period_col] = grouped_df[period_col].astype(str)
        selected_df = grouped_df.copy()
        date_col_sel = period_col
        st.success(f"✅ Data aggregated {time_period.lower()} for forecasting and visualization.")
except Exception as e:
    st.warning(f"⚠️ Could not process date grouping: {e}")



st.subheader("🔮 Forecasting (optional)") if date_col_sel and amount_col_sel: try: forecast_df = selected_df[[date_col_sel, amount_col_sel]].copy() forecast_df[date_col_sel] = pd.to_datetime(forecast_df[date_col_sel], errors="coerce") forecast_df[amount_col_sel] = pd.to_numeric(forecast_df[amount_col_sel], errors="coerce") forecast_df = forecast_df.dropna()

freq_str = "M" if "Year_Month" in forecast_df.columns else "Y" if "Year" in forecast_df.columns else "M"
    forecast_df = forecast_df.rename(columns={date_col_sel: "ds", amount_col_sel: "y"})

    if len(forecast_df) >= 3:
        horizon = st.slider("Forecast Horizon", 3, 24 if freq_str == "M" else 10, 6 if freq_str == "M" else 3)

        with st.spinner("🔄 Running forecast model..."):
            model = Prophet()
            model.fit(forecast_df)
            future = model.make_future_dataframe(periods=horizon, freq=freq_str)
            forecast = model.predict(future)

        last_date = forecast_df["ds"].max()
        hist_forecast = forecast[forecast["ds"] <= last_date]
        future_forecast = forecast[forecast["ds"] > last_date]

        st.write("### 📊 Forecast Plot")
        fig_forecast = px.line(hist_forecast, x="ds", y="yhat", title="Forecast Analysis")
        fig_forecast.add_scatter(x=future_forecast["ds"], y=future_forecast["yhat"], mode="lines", name="Forecast", line=dict(color="orange", dash="dash"))

        if show_confidence:
            fig_forecast.add_scatter(x=forecast["ds"], y=forecast["yhat_upper"], mode="lines", name="Upper Bound", line=dict(dash="dot", color="green"))
            fig_forecast.add_scatter(x=forecast["ds"], y=forecast["yhat_lower"], mode="lines", name="Lower Bound", line=dict(dash="dot", color="red"))

        fig_forecast.add_vrect(x0=last_date, x1=forecast["ds"].max(), fillcolor=forecast_color, opacity=forecast_opacity, line_width=0)
        st.plotly_chart(fig_forecast, use_container_width=True)

        forecast_table = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(horizon).copy()
        forecast_table.columns = ["Date", "Predicted", "Lower Bound", "Upper Bound"]
        st.subheader("📅 Forecast Table")
        st.dataframe(forecast_table, use_container_width=True)
    else:
        st.warning("⚠️ Not enough data points for forecasting.")
except Exception as e:
    st.error(f"❌ Forecasting failed: {e}")

else: st.info("ℹ️ Forecasting not available for this table (no Date column).")

