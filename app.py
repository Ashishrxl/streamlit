import streamlit as st
import pandas as pd
import plotly.express as px
from prophet import Prophet

# Streamlit page config
st.set_page_config(page_title="CSV Visualizer & Forecaster", layout="wide")
st.title("📊 CSV Data Visualizer with Forecasting (Interactive)")

# File uploader
uploaded_file = st.file_uploader("Upload your CSV file (joined table)", type=["csv"])

if uploaded_file is not None:
    # Read CSV without header and rename columns
    uploaded_df = pd.read_csv(uploaded_file)
    st.success("✅ File uploaded and columns renamed successfully!")

    # Show uploaded table
    st.subheader("🔍 Uploaded Table Preview (First 20 Rows)")
    st.dataframe(uploaded_df.head(20))
    with st.expander("📖 Show full uploaded table"):
        st.dataframe(uploaded_df)

    # --- Split into 3 original tables ---
    party_df = uploaded_df[["ID", "Name"]].drop_duplicates().reset_index(drop=True)
    bill_df = uploaded_df[["Bill", "PartyId", "Date", "Amount"]].drop_duplicates().reset_index(drop=True)
    billdetails_df = uploaded_df[["IndexId", "Billindex", "Item", "Qty", "Rate", "Less"]].drop_duplicates().reset_index(drop=True)

    # --- Additional joined tables ---
    party_bill_df = pd.merge(
        party_df, bill_df,
        left_on="id", right_on="partyid",
        how="inner",
        suffixes=("_party", "_bill")
    )

    bill_billdetails_df = pd.merge(
        bill_df, billdetails_df,
        left_on="bill", right_on="billindex",
        how="inner",
        suffixes=("_bill", "_details")
    )

    # --- Show all tables with first 20 rows and expanders ---
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
        st.dataframe(table_df.head(20))
        with st.expander(f"📖 Show full {table_name} Table"):
            st.dataframe(table_df)

    # --- Table selection for visualization ---
    st.subheader("📌 Select Table for Visualization")
    selected_table_name = st.selectbox("Select one table", list(tables_dict.keys()))
    selected_df = tables_dict[selected_table_name]

    st.write(f"Selected Table: **{selected_table_name}** (First 20 Rows)")
    st.dataframe(selected_df.head(20))

    # Detect categorical vs numerical for selected table
    categorical_cols = selected_df.select_dtypes(include=["object", "category"]).columns.tolist()
    numerical_cols = selected_df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    st.write("**Categorical columns:**", categorical_cols if categorical_cols else "None")
    st.write("**Numerical columns:**", numerical_cols if numerical_cols else "None")

    # Column selection for visualization based on selected table
    st.subheader("📌 Column Selection for Visualization")
    all_columns = selected_df.columns.tolist()
    selected_columns = st.multiselect(
        "Select columns to include in visualization",
        all_columns,
        default=all_columns[:5]
    )

    if selected_columns:
        st.write("Filtered Data (First 20 Rows):")
        st.dataframe(selected_df[selected_columns].head(20))
        with st.expander("📖 Show full filtered data"):
            st.dataframe(selected_df[selected_columns])

        # Visualization options
        st.subheader("📈 Interactive Visualization")
        chart_type = st.selectbox(
            "Select Chart Type",
            ["Scatter Plot", "Line Chart", "Bar Chart", "Histogram", "Correlation Heatmap"]
        )

        if chart_type == "Scatter Plot" and len(numerical_cols) >= 2:
            x_axis = st.selectbox("Select X-axis", numerical_cols)
            y_axis = st.selectbox("Select Y-axis", numerical_cols)
            fig = px.scatter(
                selected_df, x=x_axis, y=y_axis,
                color=categorical_cols[0] if categorical_cols else None
            )
            fig.update_traces(hovertemplate='X: %{x}<br>Y: %{y}<extra></extra>')
            fig.update_yaxes(tickformat=",.0f")
            st.plotly_chart(fig, use_container_width=True)

        elif chart_type == "Line Chart" and len(numerical_cols) >= 1:
            x_axis = st.selectbox("Select X-axis", all_columns)
            y_axis = st.selectbox("Select Y-axis", numerical_cols)
            fig = px.line(
                selected_df, x=x_axis, y=y_axis,
                color=categorical_cols[0] if categorical_cols else None
            )
            fig.update_traces(hovertemplate='X: %{x}<br>Y: %{y}<extra></extra>')
            fig.update_yaxes(tickformat=",.0f")
            st.plotly_chart(fig, use_container_width=True)

        elif chart_type == "Bar Chart" and categorical_cols and numerical_cols:
            x_axis = st.selectbox("Select X-axis (categorical)", categorical_cols)
            y_axis = st.selectbox("Select Y-axis (numerical)", numerical_cols)
            fig = px.bar(
                selected_df, x=x_axis, y=y_axis,
                color=categorical_cols[0] if categorical_cols else None
            )
            fig.update_traces(hovertemplate='%{x}<br>%{y}<extra></extra>')
            fig.update_yaxes(tickformat=",.0f")
            st.plotly_chart(fig, use_container_width=True)

        elif chart_type == "Histogram" and numerical_cols:
            hist_col = st.selectbox("Select column for histogram", numerical_cols)
            fig = px.histogram(selected_df, x=hist_col, nbins=30)
            fig.update_traces(hovertemplate='%{x}<br>Count: %{y}<extra></extra>')
            fig.update_yaxes(tickformat=",.0f")
            st.plotly_chart(fig, use_container_width=True)

        elif chart_type == "Correlation Heatmap" and len(numerical_cols) > 1:
            corr = selected_df[numerical_cols].corr()
            fig = px.imshow(corr, text_auto=True, color_continuous_scale="RdBu_r")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("⚠️ Not enough suitable columns for this chart.")

        # --- Forecasting Section (only if 'date' and 'amount' exist in selected table) ---
        if 'date' in selected_df.columns and 'amount' in selected_df.columns:
            try:
                selected_df['date'] = pd.to_datetime(selected_df['date'])
            except Exception as e:
                st.error(f"❌ Cannot convert 'date' column to datetime: {e}")

            forecast_df = selected_df[['date', 'amount']].dropna()
            forecast_df = forecast_df.rename(columns={'date': 'ds', 'amount': 'y'})
            forecast_df = forecast_df.groupby(pd.Grouper(key='ds', freq='M')).sum().reset_index()

            if not forecast_df.empty:
                model = Prophet()
                model.fit(forecast_df)

                # Forecast next 2 months
                future = model.make_future_dataframe(periods=2, freq='M')
                forecast = model.predict(future)

                # Interactive forecast plot using Plotly
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
                fig_forecast.update_traces(hovertemplate='Date: %{x}<br>Amount: %{y}<extra></extra>')
                fig_forecast.update_yaxes(tickformat=",.0f")
                st.plotly_chart(fig_forecast, use_container_width=True)

                # Show forecast table (last actual + 2 predicted months)
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
                st.warning("⚠️ Not enough data for forecasting.")
        else:
            st.info("ℹ️ 'date' or 'amount' column missing for forecasting.")
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