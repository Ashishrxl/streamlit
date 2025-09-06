import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from prophet import Prophet

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

    # --- Build derived tables (case-insensitive column discovery) ---
    # Party table
    id_col = find_col_ci(uploaded_df, "ID")
    name_col = find_col_ci(uploaded_df, "Name")
    if id_col and name_col:
        party_df = uploaded_df[[id_col, name_col]].drop_duplicates().reset_index(drop=True)
    else:
        party_df = pd.DataFrame()

    # Bill table
    bill_col = find_col_ci(uploaded_df, "Bill")
    partyid_col = find_col_ci(uploaded_df, "PartyId")
    date_col_master = find_col_ci(uploaded_df, "Date")
    amount_col_master = find_col_ci(uploaded_df, "Amount")
    if bill_col and partyid_col and date_col_master and amount_col_master:
        bill_df = uploaded_df[[bill_col, partyid_col, date_col_master, amount_col_master]].drop_duplicates().reset_index(drop=True)
    else:
        # Try to salvage with whatever is present
        cols_present = [c for c in [bill_col, partyid_col, date_col_master, amount_col_master] if c]
        if cols_present:
            bill_df = uploaded_df[cols_present].drop_duplicates().reset_index(drop=True)
        else:
            bill_df = pd.DataFrame()

    # BillDetails table
    indexid_col = find_col_ci(uploaded_df, "IndexId")
    billindex_col = find_col_ci(uploaded_df, "Billindex")
    item_col = find_col_ci(uploaded_df, "Item")
    qty_col = find_col_ci(uploaded_df, "Qty")
    rate_col = find_col_ci(uploaded_df, "Rate")
    less_col = find_col_ci(uploaded_df, "Less")
    billdetails_cols = [c for c in [indexid_col, billindex_col, item_col, qty_col, rate_col, less_col] if c]
    if billdetails_cols:
        billdetails_df = uploaded_df[billdetails_cols].drop_duplicates().reset_index(drop=True)
    else:
        billdetails_df = pd.DataFrame()

    # Merges (only if the required frames/cols exist)
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

    # --- Column selection (user chooses which columns to include for viz) ---
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

    # Work with the filtered dataframe for visualization
    df_vis = selected_df[selected_columns].copy()

    # Recompute categorical and numerical columns from the filtered dataframe
    categorical_cols = df_vis.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    numerical_cols = df_vis.select_dtypes(include=[np.number]).columns.tolist()

    st.write("**Categorical columns:**", categorical_cols if categorical_cols else "None")
    st.write("**Numerical columns:**", numerical_cols if numerical_cols else "None")

    st.write("Filtered Data (First 20 Rows):")
    st.dataframe(df_vis.head(20))
    with st.expander("📖 Show full filtered data"):
        st.dataframe(df_vis)

    # --- Visualization ---
    st.subheader("📈 Interactive Visualization")

    chart_type = st.selectbox(
        "Select Chart Type",
        ["Scatter Plot", "Line Chart", "Bar Chart", "Histogram", "Correlation Heatmap"]
    )

    # Make keys unique per table so reruns don't clash
    widget_key_base = selected_table_name.replace(" ", "_")

    # Scatter
    if chart_type == "Scatter Plot":
        if len(numerical_cols) < 2:
            st.warning("⚠️ Need at least two numerical columns for a scatter plot.")
        else:
            x_axis = st.selectbox("Select X-axis (numerical)", numerical_cols, index=0, key=f"scatter_x_{widget_key_base}")
            # Provide a default y different from x if possible
            default_y_idx = 1 if len(numerical_cols) > 1 and numerical_cols[1] != x_axis else 0
            # ensure y options exclude the selected x (optional)
            y_options = [c for c in numerical_cols]  # keep all options (user might want same)
            y_axis = st.selectbox("Select Y-axis (numerical)", y_options, index=default_y_idx, key=f"scatter_y_{widget_key_base}")

            color_col = categorical_cols[0] if categorical_cols else None
            try:
                hovertemplate = f"{x_axis}: " + "%{x}<br>" + f"{y_axis}: " + "%{y:,.0f}<extra></extra>"
                fig = px.scatter(df_vis, x=x_axis, y=y_axis, color=color_col)
                fig.update_traces(hovertemplate=hovertemplate)
                fig.update_yaxes(tickformat=",.0f")
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"❌ Error creating scatter plot: {e}")

    # Line chart
    elif chart_type == "Line Chart":
        if len(numerical_cols) < 1:
            st.warning("⚠️ Need at least one numerical column for a line chart.")
        else:
            x_axis = st.selectbox("Select X-axis", df_vis.columns.tolist(), key=f"line_x_{widget_key_base}")
            y_axis = st.selectbox("Select Y-axis (numerical)", numerical_cols, key=f"line_y_{widget_key_base}")
            color_col = categorical_cols[0] if categorical_cols else None
            try:
                hovertemplate = f"{x_axis}: " + "%{x}<br>" + f"{y_axis}: " + "%{y:,.0f}<extra></extra>"
                fig = px.line(df_vis, x=x_axis, y=y_axis, color=color_col)
                fig.update_traces(hovertemplate=hovertemplate)
                fig.update_yaxes(tickformat=",.0f")
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"❌ Error creating line chart: {e}")

    # Bar chart
    elif chart_type == "Bar Chart":
        if not categorical_cols or not numerical_cols:
            st.warning("⚠️ Need at least one categorical column and one numerical column for a bar chart.")
        else:
            x_axis = st.selectbox("Select X-axis (categorical)", categorical_cols, key=f"bar_x_{widget_key_base}")
            y_axis = st.selectbox("Select Y-axis (numerical)", numerical_cols, key=f"bar_y_{widget_key_base}")
            color_col = categorical_cols[0] if categorical_cols else None
            try:
                hovertemplate = f"{x_axis}: " + "%{x}<br>" + f"{y_axis}: " + "%{y:,.0f}<extra></extra>"
                fig = px.bar(df_vis, x=x_axis, y=y_axis, color=color_col)
                fig.update_traces(hovertemplate=hovertemplate)
                fig.update_yaxes(tickformat=",.0f")
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"❌ Error creating bar chart: {e}")

    # Histogram
    elif chart_type == "Histogram":
        if not numerical_cols:
            st.warning("⚠️ Need at least one numerical column for a histogram.")
        else:
            hist_col = st.selectbox("Select column for histogram", numerical_cols, key=f"hist_{widget_key_base}")
            try:
                hovertemplate = f"{hist_col}: " + "%{x}<br>" + "Count: " + "%{y:,.0f}<extra></extra>"
                fig = px.histogram(df_vis, x=hist_col, nbins=30)
                fig.update_traces(hovertemplate=hovertemplate)
                fig.update_yaxes(tickformat=",.0f")
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"❌ Error creating histogram: {e}")

    # Correlation heatmap
    elif chart_type == "Correlation Heatmap":
        if len(numerical_cols) <= 1:
            st.warning("⚠️ Need more than one numerical column for a correlation heatmap.")
        else:
            try:
                corr = df_vis[numerical_cols].corr()
                fig = px.imshow(corr, text_auto=True)
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"❌ Error creating correlation heatmap: {e}")

    # --- Forecasting: requires 'date' and 'amount' present in the filtered dataset ---
    st.subheader("🔮 Forecasting (optional)")
    # We only attempt forecasting if the user included those columns in their selected_columns
    date_col = find_col_ci(df_vis, "date")
    amount_col = find_col_ci(df_vis, "amount")

    if date_col and amount_col:
        try:
            df_vis[date_col] = pd.to_datetime(df_vis[date_col])
        except Exception as e:
            st.error(f"❌ Cannot convert '{date_col}' to datetime: {e}")
            date_col = None

    if date_col and amount_col:
        # Prepare forecast dataframe
        forecast_df = df_vis[[date_col, amount_col]].copy()
        forecast_df[amount_col] = pd.to_numeric(forecast_df[amount_col], errors="coerce")
        forecast_df = forecast_df.dropna(subset=[date_col, amount_col])

        if forecast_df.empty:
            st.warning("⚠️ Not enough data for forecasting after dropping NA.")
        else:
            # rename for Prophet
            forecast_df = forecast_df.rename(columns={date_col: "ds", amount_col: "y"})
            # aggregate monthly (user-friendly)
            try:
                forecast_df = forecast_df.groupby(pd.Grouper(key="ds", freq="M")).sum(numeric_only=True).reset_index()
            except Exception:
                # fallback: no aggregation
                forecast_df = forecast_df.sort_values("ds").reset_index(drop=True)

            if len(forecast_df) < 3:
                st.warning("⚠️ Need at least 3 monthly data points for a reliable forecast. Found: " + str(len(forecast_df)))
            else:
                try:
                    model = Prophet()
                    model.fit(forecast_df)
                    future = model.make_future_dataframe(periods=3, freq="M")
                    forecast = model.predict(future)

                    st.write("### Forecast Plot")
                    # show yhat with bounds
                    fig_forecast = px.line(
                        forecast, x="ds", y="yhat",
                        labels={"ds": "Date", "yhat": "Predicted Amount"},
                        title="Forecast of Amount"
                    )
                    fig_forecast.add_scatter(x=forecast["ds"], y=forecast["yhat_upper"], mode="lines", name="Upper Bound", line=dict(dash="dot"))
                    fig_forecast.add_scatter(x=forecast["ds"], y=forecast["yhat_lower"], mode="lines", name="Lower Bound", line=dict(dash="dot"))
                    # Use generic hovertemplate (Plotly placeholders are literal in this string)
                    fig_forecast.update_traces(hovertemplate="%{x}<br>%{y:,.0f}")
                    fig_forecast.update_yaxes(tickformat=",.0f")
                    st.plotly_chart(fig_forecast, use_container_width=True)

                    st.subheader("📅 Forecast Table (last 3 rows)")
                    forecast_table = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(3).rename(
                        columns={"ds": "Date", "yhat": "Predicted", "yhat_lower": "Lower Bound", "yhat_upper": "Upper Bound"}
                    )
                    st.dataframe(forecast_table)
                except Exception as e:
                    st.error(f"❌ Forecasting failed: {e}")
    else:
        st.info("ℹ️ To enable forecasting, include columns named 'Date' (or 'date') and 'Amount' (or 'amount') in the column selection above.")

# --- Hide Streamlit's default menu, footer, and header ---
hide_streamlit_style = """
    <style>
    #MainMenu {visibility: hidden;}     
    footer {visibility: hidden;}        
    header {visibility: hidden;}        
    </style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)