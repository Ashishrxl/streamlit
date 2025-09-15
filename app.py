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
    st.info("Upload a CSV to start. The app will derive tables and let you visualize/forecast.")
    st.stop()

try:
    file_name = uploaded_file.name
    uploaded_df = pd.read_csv(uploaded_file, low_memory=False)
except Exception as e:
    st.error(f"❌ Error reading CSV: {e}")
    st.stop()

st.success("✅ File uploaded successfully!")

# --- Conditional Logic for Column Renaming ---
proceed_with_app = False
process_alldata = False
if file_name.lower() == 'alldata.csv':
    proceed_with_app = True
    process_alldata = True
    st.info("File 'alldata.csv' detected. Proceeding with default column names.")
else:
    st.warning(f"File '{file_name}' is not 'alldata.csv'. Please confirm or rename columns.")
    st.write("### Current Columns:")
    st.dataframe(pd.DataFrame({"Original Column": uploaded_df.columns.tolist()}))

    col_check_container = st.container()
    with col_check_container:
        col1, col2 = st.columns(2)
        with col1:
            if st.button("✅ These column names are correct"):
                proceed_with_app = True
                st.session_state['column_renamed'] = False
        with col2:
            if st.button("✍️ I need to rename columns"):
                st.session_state['column_renamed'] = True

    if get_session_state('column_renamed', False):
        st.write("---")
        st.subheader("✍️ Rename Columns")
        renamed_cols = {}
        original_cols = uploaded_df.columns.tolist()
        
        rename_cols_container = st.container()
        with rename_cols_container:
            for col in original_cols:
                new_name = st.text_input(f"Rename '{col}' to:", value=col, key=f"rename_{col}")
                renamed_cols[col] = new_name
        
        if st.button("✨ Apply Renaming and Continue"):
            uploaded_df.rename(columns=renamed_cols, inplace=True)
            st.success("Columns renamed successfully! Proceeding with the app.")
            st.session_state['column_renamed'] = False
            proceed_with_app = True
    
    if not proceed_with_app:
        st.stop()
# --- End of Conditional Logic ---

# --- The rest of the app logic, which is now inside the 'if proceed_with_app' block ---
if proceed_with_app:
    if process_alldata:
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
            st.button(btn_label, key=f"btn{table_name}", on_click=toggle_state, args=(state_key,))

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
    else:
        st.subheader("🗂️ Tables Preview")
        tables_dict = {
            "Uploaded Table": uploaded_df,
        }
        state_key = "expand_UploadedTable"
        if state_key not in st.session_state:
            st.session_state[state_key] = False
        btn_label = f"Minimise Uploaded Table" if st.session_state[state_key] else f"Expand Uploaded Table"
        st.button(btn_label, key="btnUploadedTable", on_click=toggle_state, args=(state_key,))

        if st.session_state[state_key]:
            st.write("### Uploaded Table (First 20 Rows)")
            st.dataframe(uploaded_df.head(20))
            with st.expander("📖 Show full Uploaded Table"):
                st.dataframe(uploaded_df)
            st.download_button(
                "⬇️ Download Uploaded Table (CSV)",
                data=convert_df_to_csv(uploaded_df),
                file_name="uploaded_table.csv",
                mime="text/csv",
            )
            st.download_button(
                "⬇️ Download Uploaded Table (Excel)",
                data=convert_df_to_excel(uploaded_df),
                file_name="uploaded_table.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )

    # --- Start of new expandable section for Visualization ---
    st.markdown("---")
    with st.expander("📊 Visualize Data", expanded=False):
        st.subheader("📌 Select Table for Visualization")
        
        if process_alldata:
            available_tables = {k: v for k, v in tables_dict.items() if not v.empty}
            if not available_tables:
                st.warning("⚠️ No usable tables could be derived from the uploaded CSV.")
                st.stop()
            selected_table_name = st.selectbox("Select one table", list(available_tables.keys()))
            selected_df = available_tables[selected_table_name].copy()
        else:
            selected_table_name = "Uploaded Table"
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
        st.button(btn_label_processed, key="btn_processed_table", on_click=toggle_state, args=(state_key_processed,))

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

        st.subheader("📈 Interactive Visualization")

        chart_options = [
            "Scatter Plot", "Line Chart", "Bar Chart", "Histogram", "Correlation Heatmap",
            "Seaborn Scatterplot", "Seaborn Boxplot", "Seaborn Violinplot", "Seaborn Pairplot",
            "Seaborn Heatmap", "Plotly Heatmap", "Treemap", "Sunburst", "Time-Series Decomposition", "Forecasting"
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
            "Time-Series Decomposition": (True, True, False),
            "Forecasting": (True, True, False)
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
            elif chart_type == "Treemap":
                path_cols = st.multiselect("Select hierarchy for Treemap", categorical_cols, help="Select columns in order of hierarchy (e.g., Year, Name)")
                if path_cols:
                    if numerical_cols:
                        values_col = st.selectbox("Select values column for Treemap", numerical_cols, help="Select the column to determine the size of the boxes")
                        fig = px.treemap(df_vis, path=path_cols, values=values_col, title="Treemap Visualization")
                    else:
                        st.warning("⚠️ Need at least one numerical column for Treemap values.")
            elif chart_type == "Sunburst":
                path_cols = st.multiselect("Select hierarchy for Sunburst", categorical_cols, help="Select columns in order of hierarchy (e.g., Year, Name)")
                if path_cols:
                    if numerical_cols:
                        values_col = st.selectbox("Select values column for Sunburst", numerical_cols, help="Select the column to determine the size of the sectors")
                        fig = px.sunburst(df_vis, path=path_cols, values=values_col, title="Sunburst Visualization")
                    else:
                        st.warning("⚠️ Need at least one numerical column for Sunburst values.")
            elif chart_type == "Seaborn Scatterplot":
                if x_col and y_col:
                    fig, ax = plt.subplots()
                    sns.scatterplot(x=x_col, y=y_col, data=df_vis, hue=hue_col, ax=ax)
                    plt.title(f"Seaborn Scatterplot: {x_col} vs {y_col}")
                    st.pyplot(fig)
                    fig_bytes = export_matplotlib_fig(fig)
                    st.download_button("⬇️ Download Plot", data=fig_bytes, file_name="seaborn_scatter.png", mime="image/png")
                    fig = None # Prevent the next block from running
                else:
                    st.warning("⚠️ Please select both X and Y axes.")
            elif chart_type == "Seaborn Boxplot":
                if x_col and y_col:
                    fig, ax = plt.subplots()
                    sns.boxplot(x=x_col, y=y_col, data=df_vis, hue=hue_col, ax=ax)
                    plt.title(f"Seaborn Boxplot: {x_col} vs {y_col}")
                    st.pyplot(fig)
                    fig_bytes = export_matplotlib_fig(fig)
                    st.download_button("⬇️ Download Plot", data=fig_bytes, file_name="seaborn_boxplot.png", mime="image/png")
                    fig = None
                else:
                    st.warning("⚠️ Please select both X and Y axes.")
            elif chart_type == "Seaborn Violinplot":
                if x_col and y_col:
                    fig, ax = plt.subplots()
                    sns.violinplot(x=x_col, y=y_col, data=df_vis, hue=hue_col, ax=ax)
                    plt.title(f"Seaborn Violinplot: {x_col} vs {y_col}")
                    st.pyplot(fig)
                    fig_bytes = export_matplotlib_fig(fig)
                    st.download_button("⬇️ Download Plot", data=fig_bytes, file_name="seaborn_violinplot.png", mime="image/png")
                    fig = None
                else:
                    st.warning("⚠️ Please select both X and Y axes.")
            elif chart_type == "Seaborn Pairplot":
                if len(numerical_cols) >= 2:
                    fig = sns.pairplot(df_vis[numerical_cols + ([hue_col] if hue_col else [])], hue=hue_col)
                    plt.suptitle("Seaborn Pairplot", y=1.02)
                    st.pyplot(fig)
                    fig_bytes = export_matplotlib_fig(fig)
                    st.download_button("⬇️ Download Plot", data=fig_bytes, file_name="seaborn_pairplot.png", mime="image/png")
                    fig = None
                else:
                    st.warning("⚠️ Need at least 2 numerical columns for Pairplot.")
            elif chart_type == "Seaborn Heatmap":
                if len(numerical_cols) >= 2:
                    corr = df_vis[numerical_cols].corr()
                    fig, ax = plt.subplots()
                    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
                    plt.title("Seaborn Correlation Heatmap")
                    st.pyplot(fig)
                    fig_bytes = export_matplotlib_fig(fig)
                    st.download_button("⬇️ Download Plot", data=fig_bytes, file_name="seaborn_heatmap.png", mime="image/png")
                    fig = None
                else:
                    st.warning("⚠️ Need at least 2 numerical columns for heatmap.")
            elif chart_type == "Time-Series Decomposition":
                if date_col_sel and y_col:
                    try:
                        df_ts = df_vis[[date_col_sel, y_col]].copy()
                        df_ts = df_ts.set_index(date_col_sel)
                        df_ts.index = pd.to_datetime(df_ts.index)
                        result = seasonal_decompose(df_ts[y_col], model='additive', period=12)
                        fig = result.plot()
                        st.pyplot(fig)
                        fig_bytes = export_matplotlib_fig(fig)
                        st.download_button("⬇️ Download Plot", data=fig_bytes, file_name="time_series_decomposition.png", mime="image/png")
                        fig = None
                    except Exception as e:
                        st.error(f"❌ Error during time-series decomposition: {e}")
                else:
                    st.warning("⚠️ Please select a valid date column for X-axis and a numerical column for Y-axis.")
            elif chart_type == "Forecasting":
                st.subheader("🔮 Forecasting with Prophet")
                if date_col_sel and y_col:
                    try:
                        forecast_df = selected_df[[date_col_sel, y_col]].copy()
                        forecast_df.columns = ['ds', 'y']
                        forecast_df['ds'] = pd.to_datetime(forecast_df['ds'])
                        
                        model = Prophet(interval_width=show_confidence)
                        model.fit(forecast_df)

                        future_periods = st.slider("Number of periods to forecast", 1, 365, 30)
                        future = model.make_future_dataframe(periods=future_periods)
                        forecast = model.predict(future)

                        fig = px.line(forecast, x="ds", y="yhat", title="Forecast Plot")
                        fig.add_scatter(x=forecast['ds'], y=forecast['yhat'], mode='lines', name='Forecast', line=dict(color=forecast_color))
                        fig.add_scatter(x=forecast_df['ds'], y=forecast_df['y'], mode='lines', name='Actual', line=dict(color='blue'))

                        if show_confidence:
                            fig.add_scatter(
                                x=forecast['ds'], 
                                y=forecast['yhat_upper'], 
                                mode='lines', 
                                name='Upper Bound', 
                                line=dict(color=forecast_color, width=0),
                                showlegend=False
                            )
                            fig.add_scatter(
                                x=forecast['ds'], 
                                y=forecast['yhat_lower'], 
                                mode='lines', 
                                name='Lower Bound', 
                                fill='tonexty',
                                line=dict(color=forecast_color, width=0), 
                                fillcolor=f'rgba({int(255 * (int(forecast_color[1:3], 16)/255))},{int(255 * (int(forecast_color[3:5], 16)/255))},{int(255 * (int(forecast_color[5:7], 16)/255))},{forecast_opacity})',
                                showlegend=False
                            )
                        
                        fig.update_layout(title_text=f"Time Series Forecast of '{y_col}'", xaxis_title="Date", yaxis_title=y_col)
                        st.plotly_chart(fig, use_container_width=True)

                        st.subheader("📈 Trend, Weekly, and Yearly Components")
                        fig2 = model.plot_components(forecast)
                        st.pyplot(fig2)
                        
                        st.subheader("🔮 Forecast Data Table")
                        st.dataframe(forecast.tail(future_periods))
                        
                        st.download_button(
                            "⬇️ Download Forecast Data (CSV)",
                            data=convert_df_to_csv(forecast),
                            file_name="forecast_data.csv",
                            mime="text/csv",
                        )
                        st.download_button(
                            "⬇️ Download Forecast Plot (PNG)",
                            data=export_plotly_fig(fig),
                            file_name="forecast_plot.png",
                            mime="image/png",
                        )
                    except Exception as e:
                        st.error(f"❌ Error during forecasting: {e}")
                else:
                    st.warning("⚠️ Please select a valid date column for X-axis and a numerical column for Y-axis to perform forecasting.")

            if fig is not None:
                st.plotly_chart(fig, use_container_width=True)
                fig_bytes = export_plotly_fig(fig)
                if fig_bytes:
                    st.download_button("⬇️ Download Plot", data=fig_bytes, file_name=f"{chart_type.lower().replace(' ', '_')}_plot.png", mime="image/png")

        except Exception as e:
            st.error(f"❌ An error occurred while generating the chart: {e}")
            st.error("Please check your column selections and data types.")
            
