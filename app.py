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
    """Find column name case-insensitively"""
    for c in df.columns:
        if c.lower() == target.lower():
            return c
    return None

def convert_df_to_csv(df: pd.DataFrame) -> bytes:
    """Convert DataFrame to CSV bytes"""
    return df.to_csv(index=False).encode("utf-8")

def convert_df_to_excel(df: pd.DataFrame) -> bytes:
    """Convert DataFrame to Excel bytes"""
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
        df.to_excel(writer, index=False, sheet_name="Sheet1")
    return buffer.getvalue()

def export_plotly_fig(fig):
    """Export Plotly figure as PNG"""
    try:
        return pio.to_image(fig, format="png", engine="kaleido")
    except Exception:
        return None

def export_matplotlib_fig(fig):
    """Export Matplotlib figure as PNG"""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    return buf.getvalue()

# Sidebar settings
st.sidebar.header("⚙️ Settings")
forecast_color = st.sidebar.color_picker("Forecast highlight color", "#FFA500")
forecast_opacity = st.sidebar.slider("Forecast highlight opacity", 0.05, 1.0, 0.12, step=0.01)
show_confidence = st.sidebar.checkbox("Show confidence interval (upper/lower bounds)", True)

# File upload
uploaded_file = st.file_uploader("Upload your CSV file (joined table)", type=["csv"])
if uploaded_file is None:
    st.info("Upload a CSV to start. The app will derive tables and let you visualize/forecast.")
    st.stop()

# Read CSV
try:
    uploaded_df = pd.read_csv(uploaded_file, low_memory=False)
except Exception as e:
    st.error(f"❌ Error reading CSV: {e}")
    st.stop()

st.success("✅ File uploaded successfully!")

# Derive tables
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

# Tables preview
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

# Table selection for visualization
st.subheader("📌 Select Table for Visualization")
available_tables = {k: v for k, v in tables_dict.items() if not v.empty}
if not available_tables:
    st.warning("⚠️ No usable tables could be derived from the uploaded CSV.")
    st.stop()

selected_table_name = st.selectbox("Select one table", list(available_tables.keys()))
selected_df = available_tables[selected_table_name].copy()

# Detect columns for date, amount, and name
date_col_sel = find_col_ci(selected_df, "date") or find_col_ci(selected_df, "Date")
amount_col_sel = find_col_ci(selected_df, "amount") or find_col_ci(selected_df, "Amount")
name_col_sel = find_col_ci(selected_df, "name") or find_col_ci(selected_df, "Name")

# Enhanced aggregation options
if date_col_sel:
    try:
        # Convert to datetime and sort
        selected_df[date_col_sel] = pd.to_datetime(selected_df[date_col_sel], errors="coerce")
        selected_df = selected_df.sort_values(by=date_col_sel).reset_index(drop=True)

        # Add 'Year_Month' period column
        selected_df['Year_Month'] = selected_df[date_col_sel].dt.to_period('M')
        
        # Identify numerical and categorical columns
        numerical_cols = selected_df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = [c for c in selected_df.columns if c not in numerical_cols + ['Year_Month', date_col_sel]]
        
        st.markdown("### 📅 Data Aggregation Options")
        
        # Aggregation option selector
        aggregation_options = ["Original Data"]
        
        if name_col_sel:
            aggregation_options.append("Monthly Aggregated by Name")
        
        if categorical_cols:
            aggregation_options.append("Monthly Aggregated by Custom Columns")
        
        aggregation_options.append("Monthly Aggregated (No Grouping)")
        
        aggregation_choice = st.radio(
            "Choose how to aggregate your data:",
            aggregation_options,
            help="Aggregation will sum numerical columns and group by selected criteria"
        )
        
        if aggregation_choice == "Monthly Aggregated by Name" and name_col_sel:
            grouped_df = selected_df.groupby(['Year_Month', name_col_sel], as_index=False)[numerical_cols].sum()
            grouped_df['Year_Month'] = grouped_df['Year_Month'].astype(str)
            selected_df = grouped_df.copy()
            date_col_sel = 'Year_Month'
            st.success(f"✅ Data grouped by month and **{name_col_sel}** with numerical values aggregated.")
            
        elif aggregation_choice == "Monthly Aggregated by Custom Columns":
            st.markdown("#### Select Columns for Grouping")
            
            # Allow user to select columns for grouping
            selected_group_cols = st.multiselect(
                "Choose columns to group by (in addition to monthly grouping):",
                categorical_cols,
                default=[],
                help="Select one or more columns to group your data by. Numerical columns will be summed."
            )
            
            if selected_group_cols:
                group_by_cols = ['Year_Month'] + selected_group_cols
                grouped_df = selected_df.groupby(group_by_cols, as_index=False)[numerical_cols].sum()
                grouped_df['Year_Month'] = grouped_df['Year_Month'].astype(str)
                selected_df = grouped_df.copy()
                date_col_sel = 'Year_Month'
                st.success(f"✅ Data grouped by month and **{', '.join(selected_group_cols)}** with numerical values aggregated.")
            else:
                # If no columns selected, fall back to monthly only
                grouped_df = selected_df.groupby('Year_Month', as_index=False)[numerical_cols].sum()
                grouped_df['Year_Month'] = grouped_df['Year_Month'].astype(str)
                selected_df = grouped_df.copy()
                date_col_sel = 'Year_Month'
                st.info("ℹ️ No grouping columns selected. Data grouped by month only.")
                
        elif aggregation_choice == "Monthly Aggregated (No Grouping)":
            grouped_df = selected_df.groupby('Year_Month', as_index=False)[numerical_cols].sum()
            grouped_df['Year_Month'] = grouped_df['Year_Month'].astype(str)
            selected_df = grouped_df.copy()
            date_col_sel = 'Year_Month'
            st.success("✅ Data aggregated by month only. All numerical values summed per month.")
            
        elif aggregation_choice == "Original Data":
            st.info("ℹ️ Using original data without aggregation.")
            
    except Exception as e:
        st.warning(f"⚠️ Could not process date grouping: {e}")
        st.error(f"Error details: {str(e)}")

# Better table display for selected table
sel_state_key = f"expand_selected_{selected_table_name.replace(' ', '_')}"
if sel_state_key not in st.session_state:
    st.session_state[sel_state_key] = False

btn_sel_label = f"Minimise {selected_table_name} Table" if st.session_state[sel_state_key] else f"Expand {selected_table_name} Table"
clicked_sel = st.button(btn_sel_label, key="btn_selected_table")
if clicked_sel:
    st.session_state[sel_state_key] = not st.session_state[sel_state_key]

if st.session_state[sel_state_key]:
    st.write(f"### {selected_table_name} Table Preview")
    st.info(f"📊 **Data shape:** {selected_df.shape[0]} rows × {selected_df.shape[1]} columns")
    
    # Show first 20 rows
    st.write("**First 20 Rows:**")
    st.dataframe(selected_df.head(20), use_container_width=True)
    
    # Expandable section for full table
    with st.expander(f"📖 Show Full {selected_table_name} Table ({len(selected_df):,} rows)"):
        st.dataframe(selected_df, use_container_width=True)
    
    # Download options
    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            f"⬇️ Download {selected_table_name} (CSV)",
            data=convert_df_to_csv(selected_df),
            file_name=f"{selected_table_name.lower().replace(' ', '_')}_processed.csv",
            mime="text/csv",
        )
    with col2:
        st.download_button(
            f"⬇️ Download {selected_table_name} (Excel)",
            data=convert_df_to_excel(selected_df),
            file_name=f"{selected_table_name.lower().replace(' ', '_')}_processed.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

# Column selection
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

# Display column information
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Total Columns", len(selected_columns))
with col2:
    st.metric("Numerical Columns", len(numerical_cols))
with col3:
    st.metric("Categorical Columns", len(categorical_cols))

with st.expander("📋 Column Details"):
    st.write("**Categorical columns:**", categorical_cols if categorical_cols else "None")
    st.write("**Numerical columns:**", numerical_cols if numerical_cols else "None")

# Interactive visualization
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

x_col = y_col = hue_col = None

if need_x:
    x_options = [c for c in df_vis.columns]
    x_col = st.selectbox("Select X Axis", x_options, key="x_axis")

if need_y:
    y_options = [c for c in df_vis.columns if (c != x_col or not need_x)]
    y_col = st.selectbox("Select Y Axis", y_options, key="y_axis")

if need_hue:
    hue_options = [c for c in df_vis.columns if c not in [x_col, y_col]]
    hue_col = st.selectbox("Select Hue/Category (optional)", ["(None)"] + hue_options, key="hue_axis")
    if hue_col == "(None)":
        hue_col = None

st.write("### Chart:")

fig = None
try:
    if chart_type == "Scatter Plot":
        fig = px.scatter(df_vis, x=x_col, y=y_col, color=hue_col if hue_col else None, title=f"Scatter Plot: {x_col} vs {y_col}")
    elif chart_type == "Line Chart":
        fig = px.line(df_vis, x=x_col, y=y_col, color=hue_col if hue_col else None, title=f"Line Chart: {x_col} vs {y_col}")
    elif chart_type == "Bar Chart":
        fig = px.bar(df_vis, x=x_col, y=y_col, color=hue_col if hue_col else None, title=f"Bar Chart: {x_col} vs {y_col}")
    elif chart_type == "Histogram":
        fig = px.histogram(df_vis, x=x_col, color=hue_col if hue_col else None, title=f"Histogram: {x_col}")
    elif chart_type == "Correlation Heatmap":
        if len(numerical_cols) >= 2:
            corr = df_vis.select_dtypes(include=[np.number]).corr()
            fig = px.imshow(corr, text_auto=True, color_continuous_scale='RdBu', aspect='auto', title="Correlation Heatmap")
        else:
            st.warning("⚠️ Need at least 2 numerical columns for correlation heatmap.")
    elif chart_type == "Seaborn Scatterplot":
        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=df_vis, x=x_col, y=y_col, hue=hue_col if hue_col else None)
        plt.title(f"Scatterplot: {x_col} vs {y_col}")
        st.pyplot(plt.gcf())
        
        png_bytes_mpl = export_matplotlib_fig(plt.gcf())
        st.download_button("⬇️ Download Chart (PNG)", data=png_bytes_mpl, file_name="seaborn_scatterplot.png", mime="image/png")
        plt.close()
    elif chart_type == "Seaborn Boxplot":
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=df_vis, x=x_col, y=y_col, hue=hue_col if hue_col else None)
        plt.title(f"Boxplot: {x_col} vs {y_col}")
        st.pyplot(plt.gcf())
        
        png_bytes_mpl = export_matplotlib_fig(plt.gcf())
        st.download_button("⬇️ Download Chart (PNG)", data=png_bytes_mpl, file_name="seaborn_boxplot.png", mime="image/png")
        plt.close()
    elif chart_type == "Seaborn Violinplot":
        plt.figure(figsize=(10, 6))
        sns.violinplot(data=df_vis, x=x_col, y=y_col, hue=hue_col if hue_col else None)
        plt.title(f"Violinplot: {x_col} vs {y_col}")
        st.pyplot(plt.gcf())
        
        png_bytes_mpl = export_matplotlib_fig(plt.gcf())
        st.download_button("⬇️ Download Chart (PNG)", data=png_bytes_mpl, file_name="seaborn_violinplot.png", mime="image/png")
        plt.close()
    elif chart_type == "Seaborn Pairplot":
        if len(numerical_cols) >= 2:
            sns.pairplot(df_vis, hue=hue_col if hue_col else None)
            st.pyplot(plt.gcf())
            
            png_bytes_mpl = export_matplotlib_fig(plt.gcf())
            st.download_button("⬇️ Download Chart (PNG)", data=png_bytes_mpl, file_name="seaborn_pairplot.png", mime="image/png")
            plt.close()
        else:
            st.warning("⚠️ Need at least 2 numerical columns for pairplot.")
    elif chart_type == "Seaborn Heatmap":
        if len(numerical_cols) >= 2:
            plt.figure(figsize=(10, 8))
            corr = df_vis.select_dtypes(include=[np.number]).corr()
            sns.heatmap(corr, annot=True, cmap="coolwarm", center=0)
            plt.title("Correlation Heatmap")
            st.pyplot(plt.gcf())
            
            png_bytes_mpl = export_matplotlib_fig(plt.gcf())
            st.download_button("⬇️ Download Chart (PNG)", data=png_bytes_mpl, file_name="seaborn_heatmap.png", mime="image/png")
            plt.close()
        else:
            st.warning("⚠️ Need at least 2 numerical columns for heatmap.")
    elif chart_type == "Plotly Heatmap":
        fig = px.density_heatmap(df_vis, x=x_col, y=y_col, title=f"Density Heatmap: {x_col} vs {y_col}")
    elif chart_type == "Treemap":
        fig = px.treemap(df_vis, path=[x_col], values=y_col, title=f"Treemap: {x_col}")
    elif chart_type == "Sunburst":
        fig = px.sunburst(df_vis, path=[x_col], values=y_col, title=f"Sunburst: {x_col}")
    elif chart_type == "Time-Series Decomposition":
        date_series = None
        if 'Year_Month' in df_vis.columns:
            date_series = pd.to_datetime(df_vis['Year_Month'], errors="coerce")
        else:
            date_series = pd.to_datetime(df_vis[x_col], errors="coerce")
        value_series = pd.to_numeric(df_vis[y_col], errors="coerce")
        df_ts = pd.DataFrame({'x': date_series, 'y': value_series}).dropna()
        df_ts = df_ts.sort_values('x')
        df_ts.set_index('x', inplace=True)
        
        if len(df_ts) >= 24:
            result = seasonal_decompose(df_ts['y'], model="additive", period=12)
            fig, axs = plt.subplots(4, 1, figsize=(12, 10))
            result.observed.plot(ax=axs[0], title="Observed")
            result.trend.plot(ax=axs[1], title="Trend")
            result.seasonal.plot(ax=axs[2], title="Seasonal")
            result.resid.plot(ax=axs[3], title="Residual")
            plt.tight_layout()
            st.pyplot(fig)
            
            png_bytes_mpl = export_matplotlib_fig(fig)
            st.download_button("⬇️ Download Decomposition (PNG)", data=png_bytes_mpl, file_name="time_series_decomposition.png", mime="image/png")
            plt.close()
        else:
            st.warning("⚠️ At least 24 data points needed for time series decomposition.")
            
    # Display Plotly charts and download options
    if fig is not None and chart_type not in [
        "Seaborn Scatterplot", "Seaborn Boxplot", "Seaborn Violinplot",
        "Seaborn Pairplot", "Seaborn Heatmap", "Time-Series Decomposition"
    ]:
        st.plotly_chart(fig, use_container_width=True)
        png_bytes = export_plotly_fig(fig)
        if png_bytes:
            st.download_button("⬇️ Download Chart (PNG)", data=png_bytes, file_name="plotly_chart.png", mime="image/png")
            
except Exception as e:
    st.error(f"⚠️ Failed to render chart: {e}")
    st.error(f"Error details: {str(e)}")

# Forecasting section
st.subheader("🔮 Forecasting (optional)")
date_col = find_col_ci(df_vis, "date") or find_col_ci(df_vis, "Year_Month")
amount_col = find_col_ci(df_vis, "amount")

if date_col and amount_col:
    try:
        forecast_df = df_vis[[date_col, amount_col]].copy()

        if date_col == 'Year_Month':
            forecast_df[date_col] = pd.to_datetime(forecast_df[date_col], errors="coerce")
        else:
            forecast_df[date_col] = pd.to_datetime(forecast_df[date_col], errors="coerce")

        forecast_df[amount_col] = pd.to_numeric(forecast_df[amount_col], errors="coerce")
        forecast_df = forecast_df.dropna(subset=[date_col, amount_col])

        # If data is not monthly aggregated, group by month
        if date_col != 'Year_Month':
            forecast_df = forecast_df.groupby(pd.Grouper(key=date_col, freq='M')).sum(numeric_only=True).reset_index()

        forecast_df = forecast_df.rename(columns={date_col: "ds", amount_col: "y"})

        if len(forecast_df) >= 3:
            st.write(f"📈 **Forecasting based on {len(forecast_df)} data points**")
            
            col1, col2 = st.columns(2)
            with col1:
                horizon = st.slider("Forecast Horizon (months)", 3, 24, 6)
            with col2:
                st.write(f"**Data range:** {forecast_df['ds'].min().strftime('%Y-%m-%d')} to {forecast_df['ds'].max().strftime('%Y-%m-%d')}")
            
            with st.spinner("🔄 Running forecast model..."):
                model = Prophet()
                model.fit(forecast_df)
                future = model.make_future_dataframe(periods=horizon, freq="M")
                forecast = model.predict(future)

            last_date = forecast_df["ds"].max()
            hist_forecast = forecast[forecast["ds"] <= last_date]
            future_forecast = forecast[forecast["ds"] > last_date]

            st.write("### 📊 Forecast Plot")
            fig_forecast = px.line(
                hist_forecast, x="ds", y="yhat", 
                labels={"ds": "Date", "yhat": "Predicted Amount"}, 
                title=f"Forecast Analysis - Next {horizon} Months"
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

            # Forecast table with better formatting
            forecast_table = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(horizon).copy()
            forecast_table.columns = ["Date", "Predicted", "Lower Bound", "Upper Bound"]
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
                
            # Summary statistics
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
            st.warning("⚠️ Need at least 3 monthly data points for forecasting.")
    except Exception as e:
        st.error(f"❌ Forecasting failed: {e}")
        st.error(f"Error details: {str(e)}")
else:
    st.info("ℹ️ To enable forecasting, include 'Date' and 'Amount' columns in your selection.")

# Footer
st.markdown("---")
st.markdown("*Built with Streamlit - CSV Visualizer & Forecasting Tool*")
