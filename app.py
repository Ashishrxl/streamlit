import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from prophet import Prophet
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
import io

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

def export_plotly_fig(fig):
    return fig.to_image(format="png")

def export_matplotlib_fig(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    return buf.getvalue()

# Sidebar for upload
st.sidebar.header("📂 Upload Data")
uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

if uploaded_file is not None:
    # Read CSV
    try:
        uploaded_df = pd.read_csv(uploaded_file, low_memory=False)
    except Exception as e:
        st.error(f"❌ Error reading CSV: {e}")
        st.stop()

    st.success("✅ File uploaded successfully!")

    # Tabs
    tab1, tab2, tab3 = st.tabs(["🔍 Data Explorer", "📈 Visualization", "🔮 Forecasting"])

    # -------------------------
    # TAB 1: Data Explorer
    # -------------------------
    with tab1:
        st.subheader("Uploaded Data Preview")
        st.dataframe(uploaded_df.head(20))

        # Export data
        st.download_button(
            "⬇️ Download Full Data (CSV)",
            data=convert_df_to_csv(uploaded_df),
            file_name="uploaded_data.csv",
            mime="text/csv",
        )
        st.download_button(
            "⬇️ Download Full Data (Excel)",
            data=convert_df_to_excel(uploaded_df),
            file_name="uploaded_data.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

        with st.expander("📖 Show full uploaded table"):
            st.dataframe(uploaded_df)

    # -------------------------
    # TAB 2: Visualization
    # -------------------------
    with tab2:
        st.subheader("📌 Column Selection")
        all_columns = uploaded_df.columns.tolist()
        default_cols = all_columns[:5]
        selected_columns = st.multiselect(
            "Select columns for visualization (include 'Date' and 'Amount' for forecasting)",
            all_columns,
            default=default_cols
        )

        if not selected_columns:
            st.warning("⚠️ Please select at least one column.")
            st.stop()

        df_vis = uploaded_df[selected_columns].copy()
        categorical_cols = df_vis.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
        numerical_cols = df_vis.select_dtypes(include=[np.number]).columns.tolist()

        st.subheader("📊 Choose Visualization Type")
        chart_type = st.selectbox(
            "Select Chart Type",
            [
                "Scatter Plot", "Line Chart", "Bar Chart", "Histogram", 
                "Correlation Heatmap", 
                "Seaborn Boxplot", "Seaborn Violinplot", "Seaborn Pairplot",
                "Treemap", "Sunburst", "Time-Series Decomposition"
            ]
        )

        fig = None  # for Plotly
        fig_matplotlib = None  # for Matplotlib/Seaborn

        # Scatter Plot
        if chart_type == "Scatter Plot" and len(numerical_cols) >= 2:
            x_axis = st.selectbox("X-axis", numerical_cols)
            y_axis = st.selectbox("Y-axis", numerical_cols)
            color_col = st.selectbox("Color (optional)", ["None"] + categorical_cols)
            color_col = None if color_col == "None" else color_col
            fig = px.scatter(df_vis, x=x_axis, y=y_axis, color=color_col)
            st.plotly_chart(fig, use_container_width=True)

        # Example Seaborn Boxplot
        elif chart_type == "Seaborn Boxplot" and categorical_cols and numerical_cols:
            x_axis = st.selectbox("X-axis (categorical)", categorical_cols)
            y_axis = st.selectbox("Y-axis (numerical)", numerical_cols)
            hue_col = st.selectbox("Hue (optional)", ["None"] + categorical_cols)
            hue_col = None if hue_col == "None" else hue_col
            fig_matplotlib, ax = plt.subplots(figsize=(8, 5))
            sns.boxplot(data=df_vis, x=x_axis, y=y_axis, hue=hue_col, ax=ax)
            st.pyplot(fig_matplotlib)

        # Treemap
        elif chart_type == "Treemap" and categorical_cols and numerical_cols:
            path_cols = st.multiselect("Hierarchy (categorical)", categorical_cols, default=categorical_cols[:1])
            value_col = st.selectbox("Value (numerical)", numerical_cols)
            if path_cols:
                fig = px.treemap(df_vis, path=path_cols, values=value_col)
                st.plotly_chart(fig, use_container_width=True)

        # Time-Series Decomposition
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
            st.download_button(
                "⬇️ Download Chart (PNG)",
                data=export_plotly_fig(fig),
                file_name="chart.png",
                mime="image/png",
            )
        if fig_matplotlib is not None:
            st.download_button(
                "⬇️ Download Chart (PNG)",
                data=export_matplotlib_fig(fig_matplotlib),
                file_name="chart.png",
                mime="image/png",
            )

    # -------------------------
    # TAB 3: Forecasting
    # -------------------------
    with tab3:
        st.subheader("🔮 Forecasting")
        date_col = find_col_ci(uploaded_df, "date")
        amount_col = find_col_ci(uploaded_df, "amount")

        if date_col and amount_col:
            try:
                forecast_df = uploaded_df[[date_col, amount_col]].copy()
                forecast_df[date_col] = pd.to_datetime(forecast_df[date_col], errors="coerce")
                forecast_df[amount_col] = pd.to_numeric(forecast_df[amount_col], errors="coerce")
                forecast_df = forecast_df.dropna()

                forecast_df = forecast_df.rename(columns={date_col: "ds", amount_col: "y"})
                forecast_df = forecast_df.groupby(pd.Grouper(key="ds", freq="M")).sum().reset_index()

                horizon = st.slider("Forecast Horizon (months)", 3, 24, 6)

                if len(forecast_df) >= 3:
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
                    st.download_button(
                        "⬇️ Download Forecast Chart (PNG)",
                        data=export_plotly_fig(fig_forecast),
                        file_name="forecast.png",
                        mime="image/png",
                    )

                    # Export forecast data
                    st.download_button(
                        "⬇️ Download Forecast Data (CSV)",
                        data=convert_df_to_csv(forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]]),
                        file_name="forecast.csv",
                        mime="text/csv",
                    )

                    # Decomposition + Forecast
                    ts_series = forecast_df.set_index("ds")["y"]
                    if len(ts_series) >= 12:
                        model_type = st.radio("Decomposition Model", ["additive", "multiplicative"], horizontal=True, key="forecast_decomp")
                        decomposition = seasonal_decompose(ts_series, model=model_type, period=12)
                        fig_decomp, axes = plt.subplots(4, 1, figsize=(12, 8), sharex=True)
                        decomposition.observed.plot(ax=axes[0], title="Observed")
                        decomposition.trend.plot(ax=axes[1], title="Trend")
                        decomposition.seasonal.plot(ax=axes[2], title="Seasonality")
                        decomposition.resid.plot(ax=axes[3], title="Residuals")
                        st.pyplot(fig_decomp)

                        # Export decomposition
                        st.download_button(
                            "⬇️ Download Decomposition Chart (PNG)",
                            data=export_matplotlib_fig(fig_decomp),
                            file_name="decomposition.png",
                            mime="image/png",
                        )
            except Exception as e:
                st.error(f"❌ Forecasting failed: {e}")
        else:
            st.info("ℹ️ Need 'Date' and 'Amount' columns for forecasting.")