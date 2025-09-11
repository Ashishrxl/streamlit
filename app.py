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

# Hide default Streamlit style
st.markdown("""
<style>  
#MainMenu, footer, header {visibility: hidden;}  
footer {display: none !important;}  
header {display: none !important;}  
[data-testid="stToolbar"] { display: none !important; }  
</style>
""", unsafe_allow_html=True)

# --- Utility functions ---
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

# --- Sidebar settings ---
st.sidebar.header("⚙️ Settings")
forecast_color = st.sidebar.color_picker("Forecast highlight color", "#FFA500")
forecast_opacity = st.sidebar.slider("Forecast highlight opacity", 0.05, 1.0, 0.12, step=0.01)
show_confidence = st.sidebar.checkbox("Show confidence interval (upper/lower bounds)", True)

# --- File upload ---
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
if uploaded_file is None:
    st.info("Upload a CSV to start. The app will detect tables.")
    st.stop()

try:
    uploaded_df = pd.read_csv(uploaded_file, low_memory=False)
except Exception as e:
    st.error(f"❌ Error reading CSV: {e}")
    st.stop()

st.success("✅ File uploaded successfully!")

# --- Table splitting (for simplicity, treat each group of non-empty rows between empty rows as table) ---
def split_tables(df: pd.DataFrame):
    """Return list of DataFrames as tables"""
    # Detect empty rows (all NaN)
    empty_idx = df[df.isna().all(axis=1)].index.tolist()
    empty_idx = [-1] + empty_idx + [len(df)]
    tables = []
    for i in range(len(empty_idx)-1):
        sub = df.iloc[empty_idx[i]+1:empty_idx[i+1]].dropna(how='all')
        if not sub.empty:
            tables.append(sub.reset_index(drop=True))
    return tables

tables = split_tables(uploaded_df)
table_names = [f"Table {i+1}" for i in range(len(tables))]
selected_table_name = st.selectbox("Select Table to Visualize", table_names)
selected_table_idx = table_names.index(selected_table_name)
df_table = tables[selected_table_idx].copy()

st.subheader(f"Preview of {selected_table_name}")
st.dataframe(df_table.head(10), use_container_width=True)

# --- Column selection ---
all_columns = df_table.columns.tolist()
selected_columns = st.multiselect("Select columns for visualization", all_columns, default=all_columns)
if not selected_columns:
    st.warning("⚠️ Select at least one column")
    st.stop()
df_vis = df_table[selected_columns].copy()

categorical_cols = df_vis.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
numerical_cols = df_vis.select_dtypes(include=[np.number]).columns.tolist()

# --- Visualization ---
st.subheader("📈 Visualization")
chart_options = ["Scatter Plot", "Line Chart", "Bar Chart", "Histogram",
                 "Correlation Heatmap", "Seaborn Scatterplot", "Seaborn Boxplot",
                 "Seaborn Violinplot", "Seaborn Pairplot", "Seaborn Heatmap",
                 "Plotly Heatmap", "Treemap", "Sunburst", "Time-Series Decomposition"]
chart_type = st.selectbox("Select Chart Type", chart_options)

x_col = y_col = hue_col = None
if chart_type not in ["Correlation Heatmap", "Seaborn Pairplot", "Seaborn Heatmap"]:
    x_col = st.selectbox("X Axis", df_vis.columns)
    y_col = st.selectbox("Y Axis", [c for c in df_vis.columns if c != x_col])
    hue_options = [c for c in df_vis.columns if c not in [x_col, y_col]]
    hue_col = st.selectbox("Hue (optional)", ["(None)"] + hue_options)
    if hue_col == "(None)":
        hue_col = None

# Render chart
st.write("### Chart:")
fig = None
try:
    if chart_type == "Scatter Plot":
        fig = px.scatter(df_vis, x=x_col, y=y_col, color=hue_col)
    elif chart_type == "Line Chart":
        fig = px.line(df_vis, x=x_col, y=y_col, color=hue_col)
    elif chart_type == "Bar Chart":
        fig = px.bar(df_vis, x=x_col, y=y_col, color=hue_col)
    elif chart_type == "Histogram":
        fig = px.histogram(df_vis, x=x_col, color=hue_col)
    elif chart_type == "Correlation Heatmap" and len(numerical_cols)>=2:
        corr = df_vis[numerical_cols].corr()
        fig = px.imshow(corr, text_auto=True, color_continuous_scale='RdBu')
    elif chart_type == "Plotly Heatmap":
        fig = px.density_heatmap(df_vis, x=x_col, y=y_col)
    elif chart_type == "Treemap":
        fig = px.treemap(df_vis, path=[x_col], values=y_col)
    elif chart_type == "Sunburst":
        fig = px.sunburst(df_vis, path=[x_col], values=y_col)
    elif chart_type == "Seaborn Scatterplot":
        fig, ax = plt.subplots()
        sns.scatterplot(data=df_vis, x=x_col, y=y_col, hue=hue_col, ax=ax)
        st.pyplot(fig)
    elif chart_type == "Seaborn Boxplot":
        fig, ax = plt.subplots()
        sns.boxplot(data=df_vis, x=x_col, y=y_col, hue=hue_col, ax=ax)
        st.pyplot(fig)
    elif chart_type == "Seaborn Violinplot":
        fig, ax = plt.subplots()
        sns.violinplot(data=df_vis, x=x_col, y=y_col, hue=hue_col, split=True if hue_col else False, ax=ax)
        st.pyplot(fig)
    elif chart_type == "Seaborn Pairplot":
        st.pyplot(sns.pairplot(df_vis))
    elif chart_type == "Seaborn Heatmap" and len(numerical_cols)>=2:
        fig, ax = plt.subplots(figsize=(10,8))
        sns.heatmap(df_vis[numerical_cols].corr(), annot=True, fmt=".2f", cmap="RdBu", ax=ax)
        st.pyplot(fig)
    elif chart_type == "Time-Series Decomposition":
        date_series = pd.to_datetime(df_vis[x_col], errors="coerce")
        value_series = pd.to_numeric(df_vis[y_col], errors="coerce")
        df_ts = pd.DataFrame({'x': date_series, 'y': value_series}).dropna().sort_values('x')
        if len(df_ts) >= 12:
            result = seasonal_decompose(df_ts['y'], model="additive", period=12)
            fig, axs = plt.subplots(4,1, figsize=(12,10))
            result.observed.plot(ax=axs[0], title="Observed")
            result.trend.plot(ax=axs[1], title="Trend")
            result.seasonal.plot(ax=axs[2], title="Seasonal")
            result.resid.plot(ax=axs[3], title="Residual")
            plt.tight_layout()
            st.pyplot(fig)
except Exception as e:
    st.error(f"Chart error: {e}")

if fig and chart_type not in ["Seaborn Scatterplot","Seaborn Boxplot","Seaborn Violinplot",
                              "Seaborn Pairplot","Seaborn Heatmap","Time-Series Decomposition"]:
    st.plotly_chart(fig, use_container_width=True)
    png_bytes = export_plotly_fig(fig)
    if png_bytes:
        st.download_button("⬇️ Download Chart (PNG)", data=png_bytes, file_name="chart.png", mime="image/png")

# --- Forecasting ---
st.subheader("🔮 Forecasting (optional)")
date_col = find_col_ci(df_vis, "date") or find_col_ci(df_vis, "Year_Month") or find_col_ci(df_vis, "Year")
amount_col = find_col_ci(df_vis, "amount")

if date_col and amount_col:
    try:
        forecast_df = df_vis[[date_col, amount_col]].copy()
        if date_col == 'Year_Month':
            forecast_df[date_col] = pd.to_datetime(forecast_df[date_col], errors="coerce")
            freq_str, period_type = "M", "months"
        elif date_col == 'Year':
            forecast_df[date_col] = pd.to_datetime(forecast_df[date_col], errors="coerce")
            freq_str, period_type = "Y", "years"
        else:
            forecast_df[date_col] = pd.to_datetime(forecast_df[date_col], errors="coerce")
            freq_str, period_type = "M", "months"

        forecast_df[amount_col] = pd.to_numeric(forecast_df[amount_col], errors="coerce")
        forecast_df = forecast_df.dropna(subset=[date_col, amount_col])
        if date_col not in ['Year_Month','Year']:
            forecast_df = forecast_df.groupby(pd.Grouper(key=date_col,freq=freq_str)).sum(numeric_only=True).reset_index()

        forecast_df = forecast_df.rename(columns={date_col:"ds", amount_col:"y"})

        if len(forecast_df)>=3:
            horizon = st.slider(f"Forecast Horizon ({period_type})", 3 if freq_str=="M" else 1, 24 if freq_str=="M" else 10, 6 if freq_str=="M" else 3)
            model = Prophet()
            model.fit(forecast_df)
            future = model.make_future_dataframe(periods=horizon, freq=freq_str)
            forecast = model.predict(future)

            last_date = forecast_df["ds"].max()
            hist_forecast = forecast[forecast["ds"]<=last_date]
            future_forecast = forecast[forecast["ds"]>last_date]

            fig_forecast = px.line(hist_forecast, x="ds", y="yhat", labels={"ds":"Date","yhat":amount_col})
            fig_forecast.update_traces(line=dict(color="blue", dash="solid"), hovertemplate="%{y:.2f}<extra></extra>")
            fig_forecast.add_scatter(x=future_forecast["ds"], y=future_forecast["yhat"], mode="lines", name="Forecast",
                                     line=dict(color="orange", dash="dash"),
                                     hovertemplate=f"Predicted {amount_col}: %{y:.2f}<extra></extra>")
            if show_confidence:
                fig_forecast.add_scatter(x=forecast["ds"], y=forecast["yhat_upper"], mode="lines", name="Upper Bound",
                                         line=dict(dash="dot", color="green"))
                fig_forecast.add_scatter(x=forecast["ds"], y=forecast["yhat_lower"], mode="lines", name="Lower Bound",
                                         line=dict(dash="dot", color="red"))
            fig_forecast.add_vrect(x0=last_date, x1=forecast["ds"].max(), fillcolor=forecast_color, opacity=forecast_opacity,
                                   line_width=0, annotation_text="Forecast Period", annotation_position="top left")
            st.plotly_chart(fig_forecast, use_container_width=True)

            # Forecast table
            forecast_table = forecast[["ds","yhat","yhat_lower","yhat_upper"]].tail(horizon)
            forecast_table.columns = ["Date","Predicted","Lower Bound","Upper Bound"]
            forecast_table["Date"] = forecast_table["Date"].dt.strftime('%Y-%m-%d') if freq_str!="Y" else forecast_table["Date"].dt.strftime('%Y')
            forecast_table[["Predicted","Lower Bound","Upper Bound"]] = forecast_table[["Predicted","Lower Bound","Upper Bound"]].round(2)
            st.subheader("📅 Forecast Table")
            st.dataframe(forecast_table, use_container_width=True)
            col1,col2 = st.columns(2)
            with col1:
                st.download_button("⬇️ Download Forecast CSV", data=convert_df_to_csv(forecast_table), file_name="forecast_predictions.csv", mime="text/csv")
            with col2:
                st.download_button("⬇️ Download Forecast Excel", data=convert_df_to_excel(forecast_table), file_name="forecast_predictions.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        else:
            st.warning(f"⚠️ At least 3 data points needed for forecasting ({period_type})")
    except Exception as e:
        st.error(f"Forecasting failed: {e}")
else:
    st.info("ℹ️ Include 'Date' and 'Amount' columns for forecasting.")