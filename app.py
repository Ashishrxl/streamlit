import streamlit as st
import pandas as pd
import plotly.express as px
from prophet import Prophet

# Streamlit page config
st.set_page_config(page_title="CSV Visualizer & Forecaster", layout="wide")
st.title("📊 CSV Data Visualizer with Forecasting (Interactive)")

# --------- helpers ---------
def find_col(df: pd.DataFrame, target: str):
    """Return the column name in df that matches target case-insensitively."""
    for c in df.columns:
        if c.lower() == target.lower():
            return c
    return None

# File uploader
uploaded_file = st.file_uploader("Upload your CSV file (joined table)", type=["csv"])

if uploaded_file is not None:
    uploaded_df = pd.read_csv(uploaded_file)
    st.success("✅ File uploaded successfully!")

    # Show uploaded table
    st.subheader("🔍 Uploaded Table Preview (First 20 Rows)")
    st.dataframe(uploaded_df.head(20))
    with st.expander("📖 Show full uploaded table"):
        st.dataframe(uploaded_df)

    # --- Split into tables ---
    # These will work if the CSV contains these columns exactly
    try:
        party_df = uploaded_df[["ID", "Name"]].drop_duplicates().reset_index(drop=True)
        bill_df = uploaded_df[["Bill", "PartyId", "Date", "Amount"]].drop_duplicates().reset_index(drop=True)
        billdetails_df = uploaded_df[["IndexId", "Billindex", "Item", "Qty", "Rate", "Less"]].drop_duplicates().reset_index(drop=True)

        # --- Additional joined tables ---
        party_bill_df = pd.merge(
            party_df, bill_df,
            left_on="ID", right_on="PartyId",
            how="inner",
            suffixes=("_party", "_bill")
        )

        bill_billdetails_df = pd.merge(
            bill_df, billdetails_df,
            left_on="Bill", right_on="Billindex",
            how="inner",
            suffixes=("_bill", "_details")
        )
    except KeyError:
        # If any expected column is missing, build what we can and inform the user
        st.warning("⚠️ Some expected columns (e.g., ID, Name, Bill, PartyId, Date, Amount, etc.) are missing. "
                   "Only tables that can be derived from your CSV will be shown.")
        party_df = pd.DataFrame()
        bill_df = pd.DataFrame()
        billdetails_df = pd.DataFrame()
        party_bill_df = pd.DataFrame()
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
        st.stop()

    selected_table_name = st.selectbox("Select one table", list(available_tables.keys()))
    selected_df = available_tables[selected_table_name]

    st.write(f"Selected Table: **{selected_table_name}** (First 20 Rows)")
    st.dataframe(selected_df.head(20))

    categorical_cols = selected_df.select_dtypes(include=["object", "category"]).columns.tolist()
    numerical_cols = selected_df.select_dtypes(include=["int64", "float64", "int32", "float32"]).columns.tolist()
    st.write("**Categorical columns:**", categorical_cols if categorical_cols else "None")
    st.write("**Numerical columns:**", numerical_cols if numerical_cols else "None")

    # Column selection
    st.subheader("📌 Column Selection for Visualization")
    all_columns = selected_df.columns.tolist()
    default_cols = all_columns[:5] if len(all_columns) >= 5 else all_columns
    selected_columns = st.multiselect(
        "Select columns to include in visualization",
        all_columns,
        default=default_cols
    )

    if selected_columns:
        st.write("Filtered Data (First 20 Rows):")
        st.dataframe(selected_df[selected_columns].head(20))
        with st.expander("📖 Show full filtered data"):
            st.dataframe(selected_df[selected_columns])

        # --- Visualization ---
        st.subheader("📈 Interactive Visualization")
        chart_type = st.selectbox(
            "Select Chart Type",
            ["Scatter Plot", "Line Chart", "Bar Chart", "Histogram", "Correlation Heatmap"]
        )

        if chart_type == "Scatter Plot" and len(numerical_cols) >= 2:
            x_axis = st.selectbox("Select X-axis", numerical_cols, key="scatter_x")
            y_axis = st.selectbox("Select Y-axis", numerical_cols, key="scatter_y")
            fig = px.scatter(
                selected_df, x=x_axis, y=y_axis,
                color=categorical_cols[0] if categorical_cols else None
            )
            fig.update_traces(hovertemplate=f'{x_axis}: %{{x}}<br>{y_axis}: %{{y:,.0f}}<extra></extra>')
            fig.update_yaxes(tickformat=",.0f")
            st.plotly_chart(fig, use_container_width=True)

        elif chart_type == "Line Chart" and len(numerical_cols) >= 1:
            x_axis = st.selectbox("Select X-axis", all_columns, key="line_x")
            y_axis = st.selectbox("Select Y-axis", numerical_cols, key="line_y")
            fig = px.line(
                selected_df, x=x_axis, y=y_axis,
                color=categorical_cols[0] if categorical_cols else None
            )
            fig.update_traces(hovertemplate=f'{x_axis}: %{{x}}<br>{y_axis}: %{{y:,.0f}}<extra></extra>')
            fig.update_yaxes(tickformat=",.0f")
            st.plotly_chart(fig, use_container_width=True)

        elif chart_type == "Bar Chart" and categorical_cols and numerical_cols:
            x_axis = st.selectbox("Select X-axis (categorical)", categorical_cols, key="bar_x")
            y_axis = st.selectbox("Select Y-axis (numerical)", numerical_cols, key="bar_y")
            fig = px.bar(
                selected_df, x=x_axis, y=y_axis,
                color=categorical_cols[0] if categorical_cols else None
            )
            fig.update_traces(hovertemplate=f'{x_axis}: %{{x}}<br>{y_axis}: %{{y:,.0f}}<extra></extra>')
            fig.update_yaxes(tickformat=",.0f")
            st.plotly_chart(fig, use_container_width=True)

        elif chart_type == "Histogram" and numerical_cols:
            hist_col = st.selectbox("Select column for histogram", numerical_cols, key="hist_col")
            fig = px.histogram(selected_df, x=hist_col, nbins=30)
            fig.update_traces(hovertemplate=f'{hist_col}: %{{x}}<br>Count: %{{y:,.0f}}<extra></extra>')
            fig.update_yaxes(tickformat=",.0f")
            st.plotly_chart(fig, use_container_width=True)

        elif chart_type == "Correlation Heatmap" and len(numerical_cols) > 1:
            corr = selected_df[numerical_cols].corr()
            fig = px.imshow(corr, text_auto=True, color_continuous_scale="RdBu_r")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("⚠️ Not enough suitable columns for this chart.")

        # --- Forecasting (case-insensitive: looks for 'date' & 'amount') ---
        date_col = find_col(selected_df, 'date')
        amount_col = find_col(selected_df, 'amount')

        if date_col and amount_col:
            try:
                selected_df[date_col] = pd.to_datetime(selected_df[date_col])
            except Exception as e:
                st.error(f"❌ Cannot convert '{date_col}' column to datetime: {e}")
                st.stop()

            forecast_df = selected_df[[date_col, amount_col]].copy()
            # Ensure numeric y
            forecast_df[amount_col] = pd.to_numeric(forecast_df[amount_col], errors='coerce')
            forecast_df = forecast_df.dropna(subset=[date_col, amount_col])

            if not forecast_df.empty:
                forecast_df = forecast_df.rename(columns={date_col: 'ds', amount_col: 'y'})
                # Monthly aggregation
                forecast_df = forecast_df.groupby(pd.Grouper(key='ds', freq='M')).sum(numeric_only=True).reset_index()

                if len(forecast_df) >= 2:
                    model = Prophet()
                    model.fit(forecast_df)

                    future = model.make_future_dataframe(periods=2, freq='M')
                    forecast = model.predict(future)

                    st.write("### Forecast Plot")
                    fig_forecast = px.line(
                        forecast, x='ds', y='yhat',
                        title='Forecast of Amount',
                        labels={'ds': 'Date', 'yhat': 'Predicted Amount'}
                    )
                    fig_forecast.add_scatter(
                        x=forecast['ds'], y=forecast['yhat_upper'],
                        mode='lines', name='Upper Bound', line=dict(dash='dot')
                    )
                    fig_forecast.add_scatter(
                        x=forecast['ds'], y=forecast['yhat_lower'],
                        mode='lines', name='Lower Bound', line=dict(dash='dot')
                    )
                    fig_forecast.update_traces(hovertemplate='Date: %{{x}}<br>Amount: %{{y:,.0f}}<extra></extra>')
                    fig_forecast.update_yaxes(tickformat=",.0f")
                    st.plotly_chart(fig_forecast, use_container_width=True)

                    st.subheader("📅 Forecast Table")
                    forecast_table = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(3)
                    forecast_table = forecast_table.rename(
                        columns={
                            'ds': 'Date',
                            'yhat': 'Predicted',
                            'yhat_lower': 'Lower Bound',
                            'yhat_upper': 'Upper Bound'
                        }
                    )
                    st.dataframe(forecast_table)
                else:
                    st.warning("⚠️ Not enough historical points after monthly aggregation for forecasting.")
            else:
                st.warning("⚠️ Not enough data for forecasting.")
        else:
            st.info("ℹ️ Could not find both 'date' and 'amount' columns (case-insensitive) in the selected table for forecasting.")
    else:
        st.warning("⚠️ Please select at least one column for visualization.")

else:
    st.info("📂 Please upload a CSV file to continue.")

# --- Hide Streamlit's default menu, footer, and header ---
hide_streamlit_style = """
    <style>
    #MainMenu {visibility: hidden;}     
    footer {visibility: hidden;}        
    header {visibility: hidden;}        
    </style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)