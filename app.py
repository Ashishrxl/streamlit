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

# --- Page Config ---
st.set_page_config(page_title="CSV Visualizer with Forecasting", layout="wide")
st.title("📊 CSV Visualizer with Forecasting (Interactive)")

# Hide Streamlit default UI
hide_streamlit_style = """
<style>
#MainMenu, footer, header {visibility: hidden;}
footer {display: none !important;}
header {display: none !important;}
[data-testid="stToolbar"] { display: none !important; }
</style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# --- Helper Functions ---
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

# --- Sidebar Settings ---
st.sidebar.header("⚙️ Settings")
forecast_color = st.sidebar.color_picker("Forecast highlight color", "#FFA500")
forecast_opacity = st.sidebar.slider("Forecast highlight opacity", 0.05, 1.0, 0.12, step=0.01)
show_confidence = st.sidebar.checkbox("Show confidence interval (upper/lower bounds)", True)

# --- File Upload ---
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
if uploaded_file is None:
    st.info("Upload a CSV to start.")
    st.stop()

try:
    uploaded_df = pd.read_csv(uploaded_file, low_memory=False)
except Exception as e:
    st.error(f"❌ Error reading CSV: {e}")
    st.stop()
st.success("✅ File uploaded successfully!")

# --- Derive Tables ---
id_col = find_col_ci(uploaded_df, "ID")
name_col = find_col_ci(uploaded_df, "Name")
party_df = uploaded_df[[id_col, name_col]].drop_duplicates().reset_index(drop=True) if id_col and name_col else pd.DataFrame()

bill_col = find_col_ci(uploaded_df, "Bill")
partyid_col = find_col_ci(uploaded_df, "PartyId")
date_col_master = find_col_ci(uploaded_df, "Date")
amount_col_master = find_col_ci(uploaded_df, "Amount")
bill_df = (uploaded_df[[bill_col, partyid_col, date_col_master, amount_col_master]]
           .drop_duplicates().reset_index(drop=True) if bill_col and partyid_col and date_col_master and amount_col_master else pd.DataFrame())

billdetails_cols = [find_col_ci(uploaded_df, c) for c in ["IndexId", "Billindex", "Item", "Qty", "Rate", "Less"]]
billdetails_cols = [c for c in billdetails_cols if c]
billdetails_df = uploaded_df[billdetails_cols].drop_duplicates().reset_index(drop=True) if billdetails_cols else pd.DataFrame()

try:
    party_bill_df = pd.merge(party_df, bill_df, left_on=id_col, right_on=partyid_col, how="inner") if not party_df.empty and not bill_df.empty else pd.DataFrame()
except:
    party_bill_df = pd.DataFrame()

try:
    billindex_col = find_col_ci(uploaded_df, "Billindex")
    bill_billdetails_df = pd.merge(bill_df, billdetails_df, left_on=bill_col, right_on=billindex_col, how="inner") if not bill_df.empty and not billdetails_df.empty else pd.DataFrame()
except:
    bill_billdetails_df = pd.DataFrame()

# --- Tables Preview ---
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
    btn_label = f"Minimise {table_name}" if st.session_state[state_key] else f"Expand {table_name}"
    clicked = st.button(btn_label, key=f"btn_{table_name}")
    if clicked:
        st.session_state[state_key] = not st.session_state[state_key]

    if st.session_state[state_key]:
        st.write(f"### {table_name} Table (First 20 Rows)")
        if not table_df.empty:
            st.dataframe(table_df.head(20))
            with st.expander(f"📖 Show full {table_name} Table"):
                st.dataframe(table_df)
            st.download_button(f"⬇️ Download {table_name} (CSV)", data=convert_df_to_csv(table_df),
                               file_name=f"{table_name.lower().replace(' ', '_')}.csv", mime="text/csv")
            st.download_button(f"⬇️ Download {table_name} (Excel)", data=convert_df_to_excel(table_df),
                               file_name=f"{table_name.lower().replace(' ', '_')}.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        else:
            st.info("ℹ️ Not available from the uploaded CSV.")

# --- Table selection for visualization ---
st.subheader("📌 Select Table for Visualization")
available_tables = {k: v for k, v in tables_dict.items() if not v.empty}
if not available_tables:
    st.warning("⚠️ No usable tables could be derived from the uploaded CSV.")
    st.stop()

selected_table_name = st.selectbox("Select one table", list(available_tables.keys()))
selected_df = available_tables[selected_table_name].copy()

# --- Detect key columns ---
date_col_sel = find_col_ci(selected_df, "date") or find_col_ci(selected_df, "Date")
amount_col_sel = find_col_ci(selected_df, "amount") or find_col_ci(selected_df, "Amount")
name_col_sel = find_col_ci(selected_df, "name") or find_col_ci(selected_df, "Name")

# --- Aggregation and Time Series ---
if date_col_sel:
    try:
        selected_df[date_col_sel] = pd.to_datetime(selected_df[date_col_sel], errors="coerce")
        selected_df = selected_df.sort_values(by=date_col_sel).reset_index(drop=True)
        selected_df['Year_Month'] = selected_df[date_col_sel].dt.to_period('M')
        selected_df['Year'] = selected_df[date_col_sel].dt.to_period('Y')

        numerical_cols = selected_df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = [c for c in selected_df.columns if c not in numerical_cols + ['Year_Month', 'Year', date_col_sel]]

        st.markdown("### 📅 Data Aggregation Options")
        time_period = st.selectbox("Choose time aggregation period:", ["No Aggregation", "Monthly", "Yearly"])

        if time_period != "No Aggregation":
            grouping_options = ["No Grouping"]
            if name_col_sel:
                grouping_options.append("Group by Name")
            if categorical_cols:
                grouping_options.append("Group by Custom Columns")
            grouping_choice = st.selectbox(f"Choose {time_period.lower()} grouping method:", grouping_options)

            period_col = 'Year_Month' if time_period == "Monthly" else 'Year'
            if grouping_choice == "Group by Name" and name_col_sel:
                grouped_df = selected_df.groupby([period_col, name_col_sel], as_index=False)[numerical_cols].sum()
                grouped_df[period_col] = grouped_df[period_col].astype(str)
                selected_df = grouped_df.copy()
                date_col_sel = period_col
                st.success(f"✅ Data aggregated {time_period.lower()} and grouped by {name_col_sel}.")
            elif grouping_choice == "Group by Custom Columns":
                selected_group_cols = st.multiselect(f"Choose columns to group by (in addition to {time_period.lower()}):", categorical_cols, default=[])
                if selected_group_cols:
                    group_by_cols = [period_col] + selected_group_cols
                    grouped_df = selected_df.groupby(group_by_cols, as_index=False)[numerical_cols].sum()
                    grouped_df[period_col] = grouped_df[period_col].astype(str)
                    selected_df = grouped_df.copy()
                    date_col_sel = period_col
                    st.success(f"✅ Data aggregated {time_period.lower()} and grouped by {', '.join(selected_group_cols)}.")
    except Exception as e:
        st.warning(f"⚠️ Aggregation failed: {e}")

# --- Visualization ---
st.subheader("📊 Data Visualization")
if numerical_cols:
    y_axis = st.selectbox("Select numeric column for Y-axis:", numerical_cols)
    if categorical_cols:
        color_col = st.selectbox("Optional: Select column for color grouping:", [None]+categorical_cols)
    else:
        color_col = None

    try:
        fig = px.line(selected_df, x=date_col_sel, y=y_axis, color=color_col, markers=True, title=f"{y_axis} over {date_col_sel}")
        st.plotly_chart(fig, use_container_width=True)
        img_bytes = export_plotly_fig(fig)
        if img_bytes:
            st.download_button("⬇️ Download Plot as PNG", data=img_bytes, file_name="plot.png", mime="image/png")
    except Exception as e:
        st.error(f"❌ Visualization failed: {e}")

# --- Prophet Forecast ---
st.subheader("📈 Forecast with Prophet")
forecast_steps = st.number_input("Forecast steps (periods):", min_value=1, max_value=365, value=30, step=1)

if date_col_sel and y_axis:
    try:
        prophet_df = selected_df[[date_col_sel, y_axis]].rename(columns={date_col_sel: "ds", y_axis: "y"})
        model = Prophet(interval_width=0.95 if show_confidence else 0.0)
        model.fit(prophet_df)
        future = model.make_future_dataframe(periods=forecast_steps, freq='D')
        forecast = model.predict(future)

        fig2 = px.line(forecast, x='ds', y='yhat', title="Prophet Forecast", labels={'ds':'Date','yhat':'Forecast'})
        if show_confidence:
            fig2.add_traces([
                px.line(forecast, x='ds', y='yhat_upper').data[0],
                px.line(forecast, x='ds', y='yhat_lower').data[0]
            ])
        st.plotly_chart(fig2, use_container_width=True)
        img_bytes2 = export_plotly_fig(fig2)
        if img_bytes2:
            st.download_button("⬇️ Download Forecast Plot as PNG", data=img_bytes2, file_name="forecast.png", mime="image/png")
    except Exception as e:
        st.error(f"❌ Forecasting failed: {e}")