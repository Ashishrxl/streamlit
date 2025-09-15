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
#MainMenu, footer {visibility: hidden;}                  
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
</style>                """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

def find_col_ci(df: pd.DataFrame, target: str):
    for c in df.columns:
        if c.lower() == target.lower():
            return c
    return None

def convert_df_to_csv(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")

def convert_df_to_excel(df: pd.DataFrame) -> bytes:
    buffer = io.BytesOfa()
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

def toggle_state(key):
    st.session_state[key] = not st.session_state[key]

def get_session_state(key, default):
    if key not in st.session_state:
        st.session_state[key] = default
    return st.session_state[key]

st.sidebar.header("⚙️ Settings")
forecast_color = st.sidebar.color_picker("Forecast highlight color", "#FFA500")
forecast_opacity = st.sidebar.slider("Forecast highlight opacity", 0.05, 1.0, 0.12, step=0.01)
show_confidence = st.sidebar.checkbox("Show confidence interval (upper/lower bounds)", True)

uploaded_file = st.file_uploader("Upload your CSV file !", type=["csv"])
if uploaded_file is None:
    st.info("Upload a CSV to start. The app will let you visualize/forecast.")
    st.stop()

try:
    file_name = uploaded_file.name
    uploaded_df = pd.read_csv(uploaded_file, low_memory=False)
except Exception as e:
    st.error(f"❌ Error reading CSV: {e}")
    st.stop()

st.success("✅ File uploaded successfully!")

proceed_with_app = False
if 'column_check_status' not in st.session_state:
    st.session_state['column_check_status'] = 'pending'

if st.session_state['column_check_status'] == 'pending':
    if file_name.lower() == 'alldata.csv':
        st.info("File 'alldata.csv' detected. Proceeding with default column names.")
        st.session_state['column_check_status'] = 'ok'
        st.rerun()

    st.warning(f"File '{file_name}' is not 'alldata.csv'. Please confirm or rename columns.")
    st.write("### Current Columns:")
    st.dataframe(pd.DataFrame({"Original Column": uploaded_df.columns.tolist()}))

    col1, col2 = st.columns(2)
    with col1:
        if st.button("✅ These column names are correct"):
            st.session_state['column_check_status'] = 'ok'
            st.rerun()
    with col2:
        if st.button("✍️ I need to rename columns"):
            st.session_state['column_check_status'] = 'rename'
            st.rerun()

elif st.session_state['column_check_status'] == 'rename':
    st.write("---")
    st.subheader("✍️ Rename Columns")
    renamed_cols = {}
    original_cols = uploaded_df.columns.tolist()
    
    rename_cols_container = st.container()
    for col in original_cols:
        new_name = st.text_input(f"Rename '{col}' to:", value=col, key=f"rename_{col}")
        renamed_cols[col] = new_name
    
    if st.button("✨ Apply Renaming and Continue"):
        uploaded_df.rename(columns=renamed_cols, inplace=True)
        st.success("Columns renamed successfully! Proceeding with the app.")
        st.session_state['uploaded_df'] = uploaded_df  # Save the renamed DF
        st.session_state['column_check_status'] = 'ok'
        st.rerun()
else:
    if 'uploaded_df' in st.session_state:
        uploaded_df = st.session_state['uploaded_df']
    
    # --- Start of new expandable section for Visualization ---
    st.markdown("---")
    with st.expander("📊 Visualize Data", expanded=True):
        selected_df = uploaded_df.copy()
        date_col_sel = find_col_ci(selected_df, "date") or find_col_ci(selected_df, "Date")
        amount_col_sel = find_col_ci(selected_df, "amount") or find_col_ci(selected_df, "Amount")
        name_col_sel = find_col_ci(selected_df, "name") or find_col_ci(selected_df, "Name")

        if date_col_sel:
            try:
                selected_df[date_col_sel] = pd.to_datetime(selected_df[date_col_sel], errors="coerce")
                selected_df = selected_df.sort_values(by=date_col_sel).reset_index(drop=True)
                selected_df['Year_Month'] = selected_df[date_col_sel].dt.to_period('M')
                selected_df['Year'] = selected_df[date_col_sel].dt.to_period('Y')
                numerical_cols = selected_df.select_dtypes(include=[np.number]).columns.tolist()
                categorical_cols = [c for c in selected_df.columns if c not in numerical_cols + ['Year_Month', 'Year', date_col_sel]]
                
                st.markdown("### 📅 Data Aggregation Options")
                time_period = st.selectbox(
                    "Choose time aggregation period:",
                    ["No Aggregation", "Monthly", "Yearly"],
                    help="Select how you want to aggregate your data over time"
                )
                if time_period != "No Aggregation":
                    grouping_options = ["No Grouping"]
                    if name_col_sel:
                        grouping_options.append("Group by Name")
                    if categorical_cols:
                        grouping_options.append("Group by Custom Columns")
                    grouping_choice = st.selectbox(
                        f"Choose {time_period.lower()} grouping method:",
                        grouping_options,
                        help=f"Select how to group your data within each {time_period.lower()} period"
                    )
                    period_col = 'Year_Month' if time_period == "Monthly" else 'Year'
                    if grouping_choice == "Group by Name" and name_col_sel:
                        grouped_df = selected_df.groupby([period_col, name_col_sel], as_index=False)[numerical_cols].sum()
                        grouped_df[period_col] = grouped_df[period_col].astype(str)
                        selected_df = grouped_df.copy()
                        date_col_sel = period_col
                        st.success(f"✅ Data aggregated {time_period.lower()} and grouped by {name_col_sel} with numerical values summed.")
                    elif grouping_choice == "Group by Custom Columns":
                        st.markdown(f"#### Select Columns for {time_period} Grouping")
                        selected_group_cols = st.multiselect(
                            f"Choose columns to group by (in addition to {time_period.lower()} grouping):",
                            categorical_cols,
                            default=[],
                            help=f"Select one or more columns to group your data by within each {time_period.lower()} period. Numerical columns will be summed."
                        )
                        if selected_group_cols:
                            group_by_cols = [period_col] + selected_group_cols
                            grouped_df = selected_df.groupby(group_by_cols, as_index=False)[numerical_cols].sum()
                            grouped_df[period_col] = grouped_df[period_col].astype(str)
                            selected_df = grouped_df.copy()
                            date_col_sel = period_col
                            st.success(f"✅ Data aggregated {time_period.lower()} and grouped by {', '.join(selected_group_cols)} with numerical values summed.")
                        else:
                            grouped_df = selected_df.groupby(period_col, as_index=False)[numerical_cols].sum()
                            grouped_df[period_col] = grouped_df[period_col].astype(str)
                            selected_df = grouped_df.copy()
                            date_col_sel = period_col
                            st.info(f"ℹ️ No grouping columns selected. Data aggregated {time_period.lower()} only.")
                    elif grouping_choice == "No Grouping":
                        grouped_df = selected_df.groupby(period_col, as_index=False)[numerical_cols].sum()
                        grouped_df[period_col] = grouped_df[period_col].astype(str)
                        selected_df = grouped_df.copy()
                        date_col_sel = period_col
                        st.success(f"✅ Data aggregated {time_period.lower()} only. All numerical values summed per {time_period.lower()} period.")
                else:
                    st.info("ℹ️ Using original data without time aggregation.")
            except Exception as e:
                st.warning(f"⚠️ Could not process date grouping: {e}")
                st.error(f"Error details: {str(e)}")

        st.subheader("📋 Processed Data Preview")
        st.dataframe(selected_df.head(20))
        
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

        try:
            fig = None
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
                    corr = df_vis[numerical_cols].corr()
                    fig = px.imshow(corr, text_auto=True, color_continuous_scale='RdBu', aspect='auto', title="Correlation Heatmap")
                else:
                    st.warning("⚠️ Need at least 2 numerical columns for correlation heatmap.")
            elif chart_type == "Plotly Heatmap":
                if len(numerical_cols) >= 2:
                    fig = px.density_heatmap(df_vis, x=x_col, y=y_col, nbinsx=20, nbinsy=20, title="Plotly Heatmap")
                else:
                    st.warning("⚠️ Need at least 2 numerical columns for heatmap.")
            elif chart_type == "Time-Series Decomposition":
                if date_col_sel and amount_col_sel and len(df_vis) > 12:
                    df_ts = df_vis[[date_col_sel, amount_col_sel]].copy()
                    df_ts = df_ts.dropna().set_index(date_col_sel)
                    result = seasonal_decompose(df_ts, model='additive', period=12)
                    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(10, 10))
                    result.observed.plot(ax=ax1, title='Original')
                    result.trend.plot(ax=ax2, title='Trend')
                    result.seasonal.plot(ax=ax3, title='Seasonal')
                    result.resid.plot(ax=ax4, title='Residual')
                    plt.tight_layout()
                    st.pyplot(fig)
                    png_bytes_mpl = export_matplotlib_fig(fig)
                    st.download_button("⬇️ Download Chart (PNG)", data=png_bytes_mpl, file_name="ts_decomposition.png", mime="image/png")
                    plt.close()
                else:
                    st.warning("⚠️ Time-Series Decomposition requires a Date column, a numerical column, and at least 12 data points.")

            if fig is not None and chart_type != "Time-Series Decomposition":        
                st.plotly_chart(fig, use_container_width=True)        
                png_bytes_plotly = export_plotly_fig(fig)        
                if png_bytes_plotly:        
                    st.download_button("⬇️ Download Chart (PNG)", data=png_bytes_plotly, file_name="plotly_chart.png", mime="image/png")        

            if chart_type.startswith("Seaborn"):                
                plt.figure(figsize=(10, 6))                
                if chart_type == "Seaborn Scatterplot":                
                    sns.scatterplot(data=df_vis, x=x_col, y=y_col, hue=hue_col if hue_col else None)                
                elif chart_type == "Seaborn Boxplot":                
                    sns.boxplot(data=df_vis, x=x_col, y=y_col, hue=hue_col if hue_col else None)                
                elif chart_type == "Seaborn Violinplot":                
                    sns.violinplot(data=df_vis, x=x_col, y=y_col, hue=hue_col if hue_col else None)                
                elif chart_type == "Seaborn Pairplot":                
                    if len(numerical_cols) >= 2:                
                        sns.pairplot(df_vis, hue=hue_col if hue_col else None)                
                elif chart_type == "Seaborn Heatmap":                
                    if len(numerical_cols) >= 2:                
                        corr = df_vis[numerical_cols].corr()                
                        sns.heatmap(corr, annot=True, cmap="coolwarm", center=0)                
                st.pyplot(plt.gcf())                
                png_bytes_mpl = export_matplotlib_fig(plt.gcf())                
                st.download_button("⬇️ Download Chart (PNG)", data=png_bytes_mpl, file_name="seaborn_chart.png", mime="image/png")                
                plt.close()

        except Exception as e:
            st.error(f"⚠️ Failed to render chart: {e}")

    # --- End of new expandable section for Visualization ---

    # --- Start of new expandable section for Forecasting ---
    st.markdown("---")
    with st.expander("🔮 Forecasting", expanded=False):
        selected_df_forecast = uploaded_df.copy()
        date_columns = [c for c in selected_df_forecast.columns if "date" in c.lower() or c.lower() in ["year_month", "year"]]
        numerical_cols = selected_df_forecast.select_dtypes(include=[np.number]).columns.tolist()

        if date_columns and numerical_cols:
            st.markdown("---")
            st.subheader("🔮 Forecasting Options")
            selected_date_col = st.selectbox("Select Date Column for Forecasting", date_columns)
            selected_amount_col = st.selectbox("Select Numerical Column for Forecasting", numerical_cols)

            if selected_date_col and selected_amount_col:            
                forecast_df = selected_df_forecast[[selected_date_col, selected_amount_col]].copy()            
                forecast_df[selected_date_col] = pd.to_datetime(forecast_df[selected_date_col], errors="coerce")            
                forecast_df[selected_amount_col] = pd.to_numeric(forecast_df[selected_amount_col], errors="coerce")            
                forecast_df = forecast_df.dropna(subset=[selected_date_col, selected_amount_col])            

                aggregation_period = st.selectbox("Select Aggregation Period", ["No Aggregation", "Monthly", "Yearly"])            

                if aggregation_period != "No Aggregation":            
                    if aggregation_period == "Monthly":            
                        forecast_df = forecast_df.groupby(pd.Grouper(key=selected_date_col, freq='M')).sum(numeric_only=True).reset_index()            
                        freq_str = "M"            
                        period_type = "months"            
                    else:            
                        forecast_df = forecast_df.groupby(pd.Grouper(key=selected_date_col, freq='Y')).sum(numeric_only=True).reset_index()            
                        freq_str = "Y"            
                        period_type = "years"            
                else:            
                    freq_str = "M"            
                    period_type = "months"            

                original_forecast_df = forecast_df.copy()

                forecast_df = forecast_df.rename(columns={selected_date_col: "ds", selected_amount_col: "y"})            

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
                        original_forecast_df, x=selected_date_col, y=selected_amount_col,            
                        labels={selected_date_col: "Date", selected_amount_col: "Actual Amount"},            
                        title=f"Forecast Analysis - Next {horizon} {period_type.title()}"            
                    )            
                    fig_forecast.update_traces(name="Historical Data", showlegend=True, line=dict(color="blue", dash="solid"))

                    fig_forecast.add_scatter(            
                        x=hist_forecast["ds"], y=hist_forecast["yhat"],            
                        mode="lines", name="Prophet Fitted", line=dict(color="lightblue", dash="dot")            
                    )

                    fig_forecast.add_scatter(            
                        x=future_forecast["ds"], y=future_forecast["yhat"],            
                        mode="lines", name="Forecast", line=dict(color="orange", dash="dash")            
                    )            

                    if show_confidence:            
                        fig_forecast.add_scatter(            
                            x=forecast["ds"], y=forecast["yhat_upper"],            
                            mode="lines", name="Upper Bound", line=dict(dash="dot", color="green"), showlegend=True            
                        )            
                        fig_forecast.add_scatter(            
                            x=forecast["ds"], y=forecast["yhat_lower"],            
                            mode="lines", name="Lower Bound", line=dict(dash="dot", color="red"), showlegend=True            
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
            st.info("ℹ️ The selected table does not contain a valid date column and/or a numerical column for forecasting.")
    # --- End of new expandable section for Forecasting ---
