import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from prophet import Prophet
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
import io
import plotly.io as pio

# Streamlit page config
st.set_page_config(page_title="CSV Visualizer & Forecaster", layout="wide")
st.title("📊 CSV Data Visualizer with Forecasting (Interactive)")

# --- Helpers ---
def find_col_ci(df: pd.DataFrame, target: str):
    """Return the actual column name in df that matches target case-insensitively, or None."""
    for c in df.columns:
        if c.lower() == target.lower():
            return c
    return None

def convert_df_to_csv(df):
    return df.to_csv(index=False).encode("utf-8")

def convert_df_to_excel(df):
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
        df.to_excel(writer, index=False, sheet_name="Sheet1")
    return buffer.getvalue()

# --- Plot export helpers ---
def export_plotly_fig(fig):
    try:
        return pio.to_image(fig, format="png", engine="kaleido")
    except Exception:
        return None

def export_plotly_html(fig):
    return fig.to_html(include_plotlyjs="cdn")

def export_matplotlib_fig(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    return buf.getvalue()

# File uploader
uploaded_file = st.file_uploader("Upload your CSV file (joined table)", type=["csv"])

if uploaded_file is not None:
    # Read CSV
    try:
        uploaded_df = pd.read_csv(uploaded_file, low_memory=False)
    except Exception as e:
        st.error(f"❌ Error reading CSV: {e}")
        st.stop()

    st.success("✅ File uploaded successfully!")

    # Show uploaded table
    st.subheader("🔍 Uploaded Table Preview (First 20 Rows)")
    st.dataframe(uploaded_df.head(20))
    with st.expander("📖 Show full uploaded table"):
        st.dataframe(uploaded_df)

    # --- Build derived tables ---
    id_col = find_col_ci(uploaded_df, "ID")
    name_col = find_col_ci(uploaded_df, "Name")
    if id_col and name_col:
        party_df = uploaded_df[[id_col, name_col]].drop_duplicates().reset_index(drop=True)
    else:
        party_df = pd.DataFrame()

    bill_col = find_col_ci(uploaded_df, "Bill")
    partyid_col = find_col_ci(uploaded_df, "PartyId")
    date_col_master = find_col_ci(uploaded_df, "Date")
    amount_col_master = find_col_ci(uploaded_df, "Amount")
    if bill_col and partyid_col and date_col_master and amount_col_master:
        bill_df = uploaded_df[[bill_col, partyid_col, date_col_master, amount_col_master]].drop_duplicates().reset_index(drop=True)
    else:
        bill_df = pd.DataFrame()

    billdetails_cols = [find_col_ci(uploaded_df, c) for c in ["IndexId", "Billindex", "Item", "Qty", "Rate", "Less"]]
    billdetails_cols = [c for c in billdetails_cols if c]
    if billdetails_cols:
        billdetails_df = uploaded_df[billdetails_cols].drop_duplicates().reset_index(drop=True)
    else:
        billdetails_df = pd.DataFrame()

    try:
        if not party_df.empty and not bill_df.empty and id_col and partyid_col:
            party_bill_df = pd.merge(
                party_df, bill_df,
                left_on=id_col, right_on=partyid_col,
                how="inner", suffixes=("_party", "_bill")
            )
        else:
            party_bill_df = pd.DataFrame()
    except Exception:
        party_bill_df = pd.DataFrame()

    try:
        billindex_col = find_col_ci(uploaded_df, "Billindex")
        if not bill_df.empty and not billdetails_df.empty and bill_col and billindex_col:
            bill_billdetails_df = pd.merge(
                bill_df, billdetails_df,
                left_on=bill_col, right_on=billindex_col,
                how="inner", suffixes=("_bill", "_details")
            )
        else:
            bill_billdetails_df = pd.DataFrame()
    except Exception:
        bill_billdetails_df = pd.DataFrame()

    # --- Show all tables ---
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
        st.write(f"### {table_name} Table (First 20 Rows)")
        if not table_df.empty:
            st.dataframe(table_df.head(20))
            with st.expander(f"📖 Show full {table_name} Table"):
                st.dataframe(table_df)

            # Download buttons for each table
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

    # --- Table selection ---
    st.subheader("📌 Select Table for Visualization")
    available_tables = {k: v for k, v in tables_dict.items() if not v.empty}
    if not available_tables:
        st.warning("⚠️ No usable tables could be derived from the uploaded CSV.")
        st.stop()

    selected_table_name = st.selectbox("Select one table", list(available_tables.keys()))
    selected_df = available_tables[selected_table_name].copy()

    st.write(f"Selected Table: **{selected_table_name}** (First 20 Rows)")
    st.dataframe(selected_df.head(20))

    # --- Column selection ---
    st.subheader("📌 Column Selection for Visualization")
    all_columns = selected_df.columns.tolist()
    default_cols = all_columns[:5] if all_columns else []
    selected_columns = st.multiselect(
        "Select columns to include in visualization (include 'Date' and 'Amount' if you want forecasting)",
        all_columns,
        default=default_cols
    )

    if not selected_columns:
        st.warning("⚠️ Please select at least one column for visualization.")
        st.stop()

    df_vis = selected_df[selected_columns].copy()
    categorical_cols = df_vis.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    numerical_cols = df_vis.select_dtypes(include=[np.number]).columns.tolist()

    st.write("**Categorical columns:**", categorical_cols if categorical_cols else "None")
    st.write("**Numerical columns:**", numerical_cols if numerical_cols else "None")

    # --- Visualization ---
    st.subheader("📈 Interactive Visualization")
    chart_type = st.selectbox(
        "Select Chart Type",
        [
            "Scatter Plot", "Line Chart", "Bar Chart", "Histogram", "Correlation Heatmap",
            "Seaborn Boxplot", "Seaborn Violinplot", "Seaborn Pairplot",
            "Treemap", "Sunburst", "Time-Series Decomposition"
        ]
    )

    fig = None
    fig_matplotlib = None

    # Scatter
    if chart_type == "Scatter Plot" and len(numerical_cols) >= 2:
        x_axis = st.selectbox("X-axis", numerical_cols)
        y_axis = st.selectbox("Y-axis", numerical_cols)
        color_col = st.selectbox("Color (optional)", ["None"] + categorical_cols)
        color_col = None if color_col == "None" else color_col
        fig = px.scatter(df_vis, x=x_axis, y=y_axis, color=color_col)
        st.plotly_chart(fig, use_container_width=True)

    elif chart_type == "Seaborn Boxplot" and categorical_cols and numerical_cols:
        x_axis = st.selectbox("X-axis (categorical)", categorical_cols)
        y_axis = st.selectbox("Y-axis (numerical)", numerical_cols)
        hue_col = st.selectbox("Hue (optional)", ["None"] + categorical_cols)
        hue_col = None if hue_col == "None" else hue_col
        fig_matplotlib, ax = plt.subplots(figsize=(8, 5))
        sns.boxplot(data=df_vis, x=x_axis, y=y_axis, hue=hue_col, ax=ax)
        st.pyplot(fig_matplotlib)

    elif chart_type == "Treemap" and categorical_cols and numerical_cols:
        path_cols = st.multiselect("Hierarchy (categorical)", categorical_cols, default=categorical_cols[:1])
        value_col = st.selectbox("Value (numerical)", numerical_cols)
        if path_cols:
            fig = px.treemap(df_vis, path=path_cols, values=value_col)
            st.plotly_chart(fig, use_container_width=True)

    elif chart_type == "Time-Series Decomposition":
        date_col = find_col_ci(df_vis, "date")
        amount_col = find_col_ci(df_vis, "amount")
        if date_col and amount_col:
            ts_df = df_vis[[date_col, amount_col]].copy()
            ts_df[date_col] = pd.to_datetime(ts_df[date_col], errors="coerce")
            ts_df[amount_col] = pd.to_numeric(ts_df[amount_col], errors="coerce")
            ts_df = ts_df.dropna().set_index(date_col).sort_index()
            if len(ts_df) >= 12:
                model_type = st.radio("Decomposition Model", ["additive", "multiplicative"], horizontal=True)
                decomposition = seasonal_decompose(ts_df[amount_col], model=model_type, period=12)
                fig_matplotlib, axes = plt.subplots(4, 1, figsize=(12, 8), sharex=True)
                decomposition.observed.plot(ax=axes[0], title="Observed")
                decomposition.trend.plot(ax=axes[1], title="Trend")
                decomposition.seasonal.plot(ax=axes[2], title="Seasonality")
                decomposition.resid.plot(ax=axes[3], title="Residuals")
                st.pyplot(fig_matplotlib)

    # --- Export buttons ---
    if fig is not None:
        png_bytes = export_plotly_fig(fig)
        if png_bytes:
            st.download_button(
                "⬇️ Download Chart (PNG)",
                data=png_bytes,
                file_name="chart.png",
                mime="image/png",
            )
        st.download_button(
            "⬇️ Download Chart (HTML, interactive)",
            data=export_plotly_html(fig),
            file_name="chart.html",
            mime="text/html",
        )

    if fig_matplotlib is not None:
        st.download_button(
            "⬇️ Download Chart (PNG)",
            data=export_matplotlib_fig(fig_matplotlib),
            file_name="chart.png",
            mime="image/png",
        )

    # --- Forecasting ---
    st.subheader("🔮 Forecasting (optional)")
    date_col = find_col_ci(df_vis, "date")
    amount_col = find_col_ci(df_vis, "amount")

    if date_col and amount_col:
        try:
            df_vis[date_col] = pd.to_datetime(df_vis[date_col])
            forecast_df = df_vis[[date_col, amount_col]].copy()
            forecast_df[amount_col] = pd.to_numeric(forecast_df[amount_col], errors="coerce")
            forecast_df = forecast_df.dropna(subset=[date_col, amount_col])

            forecast_df = forecast_df.rename(columns={date_col: "ds", amount_col: "y"})
            forecast_df = forecast_df.groupby(pd.Grouper(key="ds", freq="M")).sum(numeric_only=True).reset_index()

            if len(forecast_df) >= 3:
                horizon = st.slider("Forecast Horizon (months)", 3, 24, 6)
                model = Prophet()
                model.fit(forecast_df)
                future = model.make_future_dataframe(periods=horizon, freq="M")
                forecast = model.predict(future)

                st.write("### Forecast Plot")
                fig_forecast = px.line(forecast, x="ds", y="yhat", labels={"ds": "Date", "yhat": "Predicted Amount"})
                fig_forecast.add_scatter(x=forecast["ds"], y=forecast["yhat_upper"], mode="lines", name="Upper Bound", line=dict(dash="dot"))
                fig_forecast.add_scatter(x=forecast["ds"], y=forecast["yhat_lower"], mode="lines", name="Lower Bound", line=dict(dash="dot"))
                st.plotly_chart(fig_forecast, use_container_width=True)

                # Export forecast plot
                png_bytes = export_plotly_fig(fig_forecast)
                if png_bytes:
                    st.download_button(
                        "⬇️ Download Forecast Chart (PNG)",
                        data=png_bytes,
                        file_name="forecast.png",
                        mime="image/png",
                    )
                st.download_button(
                    "⬇️ Download Forecast Chart (HTML, interactive)",
                    data=export_plotly_html(fig_forecast),
                    file_name="forecast.html",
                    mime="text/html",
                )

                # Forecast table
                st.subheader("📅 Forecast Table (last few rows)")
                forecast_table = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(6).rename(
                    columns={"ds": "Date", "yhat": "Predicted", "yhat_lower": "Lower Bound", "yhat_upper": "Upper Bound"}
                )
                st.dataframe(forecast_table)

                # Export forecast data
                st.download_button(
                    "⬇️ Download Forecast Data (CSV)",
                    data=convert_df_to_csv(forecast_table),
                    file_name="forecast.csv",
                    mime="text/csv",
                )

        except Exception as e:
            st.error(f"❌ Forecasting failed: {e}")
    else:
        st.info("ℹ️ To enable forecasting, include 'Date' and 'Amount' columns in your selection.")

# --- Hide Streamlit's default menu, footer, and header ---
hide_streamlit_style = """
    <style>
    #MainMenu {visibility: hidden;}     
    footer {visibility: hidden;}        
    header {visibility: hidden;}        
    </style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)