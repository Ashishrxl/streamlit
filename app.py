import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from prophet import Prophet

# Streamlit page config
st.set_page_config(page_title="CSV Data Visualizer & Forecaster", layout="wide")
st.title("📊 CSV Data Visualizer with Forecasting")

# File uploader
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    # Read CSV without header
    df = pd.read_csv(uploaded_file, header=None)
    
    # Rename columns
    df.columns = ["id", "name", "indexid", "billindex", "item", "qty", "rate", 
                  "less", "bill", "partyid", "date", "amount"]
    
    st.success("✅ File uploaded and columns renamed successfully!")
    
    # Data preview
    st.subheader("🔍 Data Preview")
    st.dataframe(df.head())

    # Detect categorical vs numerical
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    numerical_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()

    st.write("**Categorical columns:**", categorical_cols if categorical_cols else "None")
    st.write("**Numerical columns:**", numerical_cols if numerical_cols else "None")

    # Column selection
    st.subheader("📌 Column Selection")
    all_columns = df.columns.tolist()
    selected_columns = st.multiselect("Select columns to include in visualization", all_columns, default=all_columns[:5])

    if selected_columns:
        st.write("Filtered Data:")
        st.dataframe(df[selected_columns].head())

        # Visualization options
        st.subheader("📈 Visualization")
        chart_type = st.selectbox("Select Chart Type", 
                                  ["Scatter Plot", "Line Chart", "Bar Chart", "Histogram", "Correlation Heatmap"])

        if chart_type == "Scatter Plot" and len(numerical_cols) >= 2:
            x_axis = st.selectbox("Select X-axis", numerical_cols)
            y_axis = st.selectbox("Select Y-axis", numerical_cols)
            fig, ax = plt.subplots()
            sns.scatterplot(data=df, x=x_axis, y=y_axis, ax=ax)
            st.pyplot(fig)

        elif chart_type == "Line Chart" and len(numerical_cols) >= 1:
            x_axis = st.selectbox("Select X-axis", all_columns)
            y_axis = st.selectbox("Select Y-axis", numerical_cols)
            fig, ax = plt.subplots()
            sns.lineplot(data=df, x=x_axis, y=y_axis, ax=ax)
            st.pyplot(fig)

        elif chart_type == "Bar Chart" and categorical_cols and numerical_cols:
            x_axis = st.selectbox("Select X-axis (categorical)", categorical_cols)
            y_axis = st.selectbox("Select Y-axis (numerical)", numerical_cols)
            fig, ax = plt.subplots()
            sns.barplot(data=df, x=x_axis, y=y_axis, ax=ax)
            st.pyplot(fig)

        elif chart_type == "Histogram" and numerical_cols:
            hist_col = st.selectbox("Select column for histogram", numerical_cols)
            fig, ax = plt.subplots()
            sns.histplot(df[hist_col], kde=True, ax=ax)
            st.pyplot(fig)

        elif chart_type == "Correlation Heatmap" and len(numerical_cols) > 1:
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.heatmap(df[numerical_cols].corr(), annot=True, cmap="coolwarm", ax=ax)
            st.pyplot(fig)
        else:
            st.warning("⚠️ Not enough suitable columns for this chart.")

        # --- Forecasting Section ---
        st.subheader("🔮 Forecasting (Next 2 Months, Monthly Sum)")

        # Ensure 'date' is datetime
        try:
            df['date'] = pd.to_datetime(df['date'])
        except Exception as e:
            st.error(f"❌ Cannot convert 'date' column to datetime: {e}")

        # Prepare data for Prophet (monthly aggregation with sum)
        if 'date' in df.columns and 'amount' in df.columns:
            forecast_df = df[['date', 'amount']].dropna()
            forecast_df = forecast_df.rename(columns={'date': 'ds', 'amount': 'y'})
            forecast_df = forecast_df.groupby(pd.Grouper(key='ds', freq='M')).sum().reset_index()

            if not forecast_df.empty:
                model = Prophet()
                model.fit(forecast_df)

                # Forecast next 2 months
                future = model.make_future_dataframe(periods=2, freq='M')
                forecast = model.predict(future)

                # Plot forecast
                st.write("### Forecast Plot")
                fig1 = model.plot(forecast)
                st.pyplot(fig1)

                st.write("### Forecast Components")
                fig2 = model.plot_components(forecast)
                st.pyplot(fig2)

                # Show forecast table (last actual + 2 predicted months)
                st.subheader("📅 Forecast Table")
                forecast_table = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(3)
                forecast_table = forecast_table.rename(columns={
                    'ds': 'Date',
                    'yhat': 'Predicted',
                    'yhat_lower': 'Lower Bound',
                    'yhat_upper': 'Upper Bound'
                })
                st.dataframe(forecast_table)
            else:
                st.warning("⚠️ Not enough data for forecasting.")
        else:
            st.info("ℹ️ 'date' or 'amount' column missing for forecasting.")
    else:
        st.warning("⚠️ Please select at least one column for visualization.")

else:
    st.info("📂 Please upload a CSV file to continue.")