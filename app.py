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
            freq_setting = "M" if time_period == "Monthly" else "Y"
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

chart_container = st.container()
with chart_container:
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
    col1, col2 = st.columns([1, 3])
    with col1:
        with st.form("chart_selection_form"):
            chart_type = st.selectbox(
                "Select Chart Type", 
                chart_options, 
                index=chart_options.index(st.session_state.chart_type_selected) if st.session_state.chart_type_selected in chart_options else 0
            )
            need_x, need_y, need_hue = chart_x_y_hue_req.get(chart_type, (True, True, False))
            x_col = y_col = hue_col = None
            if need_x:
                x_options = [c for c in df_vis.columns]
                default_x_index = 0
                if st.session_state.x_col_selected and st.session_state.x_col_selected in x_options:
                    default_x_index = x_options.index(st.session_state.x_col_selected)
                x_col = st.selectbox("Select X Axis", x_options, index=default_x_index)
            if need_y:
                y_options = [c for c in df_vis.columns if (c != x_col or not need_x)]
                default_y_index = 0
                if st.session_state.y_col_selected and st.session_state.y_col_selected in y_options:
                    default_y_index = y_options.index(st.session_state.y_col_selected)
                elif len(y_options) > 1:
                    default_y_index = 1
                y_col = st.selectbox("Select Y Axis", y_options, index=default_y_index)
            if need_hue:
                hue_options = [c for c in df_vis.columns if c not in [x_col, y_col]]
                hue_full_options = ["(None)"] + hue_options
                default_hue_index = 0
                if st.session_state.hue_col_selected and st.session_state.hue_col_selected in hue_full_options:
                    default_hue_index = hue_full_options.index(st.session_state.hue_col_selected)
                hue_col = st.selectbox("Select Hue/Category (optional)", hue_full_options, index=default_hue_index)
                if hue_col == "(None)":
                    hue_col = None
            submitted = st.form_submit_button("Generate Chart")
            if submitted:
                st.session_state.chart_type_selected = chart_type
                st.session_state.x_col_selected = x_col
                st.session_state.y_col_selected = y_col
                st.session_state.hue_col_selected = hue_col
                st.session_state.chart_rendered = True
    with col2:
        chart_display_container = st.container()
        if st.session_state.chart_rendered or submitted:
            with chart_display_container:
                st.write("### Chart:")
                try:
                    fig = None
                    chart_type_to_render = st.session_state.chart_type_selected
                    x_col_to_render = st.session_state.x_col_selected
                    y_col_to_render = st.session_state.y_col_selected
                    hue_col_to_render = st.session_state.hue_col_selected
                    if chart_type_to_render == "Scatter Plot":
                        fig = px.scatter(df_vis, x=x_col_to_render, y=y_col_to_render, color=hue_col_to_render if hue_col_to_render else None, title=f"Scatter Plot: {x_col_to_render} vs {y_col_to_render}")
                    elif chart_type_to_render == "Line Chart":
                        fig = px.line(df_vis, x=x_col_to_render, y=y_col_to_render, color=hue_col_to_render if hue_col_to_render else None, title=f"Line Chart: {x_col_to_render} vs {y_col_to_render}")
                    elif chart_type_to_render == "Bar Chart":
                        fig = px.bar(df_vis, x=x_col_to_render, y=y_col_to_render, color=hue_col_to_render if hue_col_to_render else None, title=f"Bar Chart: {x_col_to_render} vs {y_col_to_render}")
                    elif chart_type_to_render == "Histogram":
                        fig = px.histogram(df_vis, x=x_col_to_render, color=hue_col_to_render if hue_col_to_render else None, title=f"Histogram: {x_col_to_render}")
                    elif chart_type_to_render == "Correlation Heatmap":
                        if len(numerical_cols) >= 2:
                            corr = df_vis[numerical_cols].corr()
                            fig = px.imshow(corr, text_auto=True, color_continuous_scale='RdBu', aspect='auto', title="Correlation Heatmap")
                        else:
                            st.warning("⚠️ Need at least 2 numerical columns for correlation heatmap.")
                    elif chart_type_to_render == "Plotly Heatmap":
                        if len(numerical_cols) >= 2:
                            fig = px.density_heatmap(df_vis, x=x_col_to_render, y=y_col_to_render, nbinsx=20, nbinsy=20, title="Plotly Heatmap")
                        else:
                            st.warning("⚠️ Need at least 2 numerical columns for heatmap.")
                    if fig is not None:            
                        st.plotly_chart(fig, use_container_width=True, key=f"plotly_chart_{chart_type_to_render}")            
                        png_bytes_plotly = export_plotly_fig(fig)            
                        if png_bytes_plotly:            
                            st.download_button("⬇️ Download Chart (PNG)", data=png_bytes_plotly, file_name="plotly_chart.png", mime="image/png")
                    if chart_type_to_render.startswith("Seaborn"):                    
                        plt.figure(figsize=(10, 6))                    
                        if chart_type_to_render == "Seaborn Scatterplot":
                            sns.scatterplot(data=df_vis, x=x_col_to_render, y=y_col_to_render, hue=hue_col_to_render if hue_col_to_render else None)
                        elif chart_type_to_render == "Seaborn Boxplot":
                            sns.boxplot(data=df_vis, x=x_col_to_render, y=y_col_to_render, hue=hue_col_to_render if hue_col_to_render else None)
                        elif chart_type_to_render == "Seaborn Violinplot":
                            sns.violinplot(data=df_vis, x=x_col_to_render, y=y_col_to_render, hue=hue_col_to_render if hue_col_to_render else None)
                        elif chart_type_to_render == "Seaborn Pairplot":
                            if len(numerical_cols) >= 2:
                                sns.pairplot(df_vis, hue=hue_col_to_render if hue_col_to_render else None)
                        elif chart_type_to_render == "Seaborn Heatmap":
                            if len(numerical_cols) >= 2:
                                corr = df_vis[numerical_cols].corr()
                                sns.heatmap(corr, annot=True, cmap="coolwarm", center=0)
                        st.pyplot(plt.gcf(), key=f"seaborn_chart_{chart_type_to_render}")
                        png_bytes_mpl = export_matplotlib_fig(plt.gcf())
                        st.download_button("⬇️ Download Chart (PNG)", data=png_bytes_mpl, file_name="seaborn_chart.png", mime="image/png")
                        plt.close()
                except Exception as e:
                    st.error(f"⚠️ Failed to render chart: {e}")
        else:
            with chart_display_container:
                st.info("Select chart options and click 'Generate Chart' to create visualization.")

st.subheader("🔮 Forecasting (optional)")
date_columns = [c for c in df_vis.columns if "date" in c.lower() or c.lower() in ["year_month", "year"]]
if date_columns and numerical_cols:
    if st.session_state.forecast_section_stage == "select_columns":
        with st.form("forecast_column_form"):
            selected_date_col = st.selectbox("Select Date Column for Forecasting", date_columns, index=0)
            selected_amount_col = st.selectbox("Select Numerical Column for Forecasting", numerical_cols, index=0)
            next_step = st.form_submit_button("Next: Select Aggregation")
        if next_step:
            st.session_state.forecast_selected_date_col = selected_date_col
            st.session_state.forecast_selected_amount_col = selected_amount_col
            st.session_state.forecast_section_stage = "aggregation"
            st.experimental_rerun()
    elif st.session_state.forecast_section_stage == "aggregation":
        with st.form("forecast_aggregation_form"):
            aggregation_period = st.selectbox("Select Aggregation Period", ["No Aggregation", "Monthly", "Yearly"], index=0)
            next_step = st.form_submit_button("Next: Run Forecast")
        if next_step:
            st.session_state.forecast_aggregation_period = aggregation_period
            st.session_state.forecast_section_stage = "forecast"
            st.experimental_rerun()
    elif st.session_state.forecast_section_stage == "forecast":
        selected_date_col = st.session_state.forecast_selected_date_col
        selected_amount_col = st.session_state.forecast_selected_amount_col
        aggregation_period = st.session_state.forecast_aggregation_period
        forecast_df = df_vis[[selected_date_col, selected_amount_col]].copy()
        forecast_df[selected_date_col] = pd.to_datetime(forecast_df[selected_date_col], errors="coerce")
        forecast_df[selected_amount_col] = pd.to_numeric(forecast_df[selected_amount_col], errors="coerce")
        forecast_df = forecast_df.dropna(subset=[selected_date_col, selected_amount_col])
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
                hist_forecast, x="ds", y="yhat",
                labels={"ds": "Date", "yhat": "Predicted Amount"},
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
            st.plotly_chart(fig_forecast, use_container_width=True, key="forecast_chart")
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
        if st.button("Start Over"):
            st.session_state.forecast_section_stage = "select_columns"
            st.experimental_rerun()
else:
    st.info("ℹ️ To enable forecasting, include a valid date column and at least one numerical column in your selection.")