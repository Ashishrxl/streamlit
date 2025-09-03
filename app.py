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
    # Read CSV
    df = pd.read_csv(uploaded_file)
    st.success("✅ File uploaded successfully!")
    
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
    selected_columns = st.multiselect("Select columns to include", all_columns, default=all_columns[:2])

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
        date_cols = df.select_dtypes(include=["datetime64[ns]"]).columns.tolist()

        # Try parsing object columns as dates
        for col in df.select_dtypes(include=["object"]).columns:
            try:
                df[col] = pd.to_datetime(df[col])
                date_cols.append(col)
            except Exception:
                continue
        date_cols = list(set(date_cols))  # unique

        if date_cols and numerical_cols:
            time_col = st.selectbox("Select Date/Time column", date_cols)
            target_col = st.selectbox("Select target column for forecasting", numerical_cols)

            # Prepare data for Prophet (monthly aggregation with sum)
            forecast_df = df[[time_col, target_col]].dropna()
            forecast_df = forecast_df.rename(columns={time_col: "ds", target_col: "y"})
            forecast_df = forecast_df.groupby(pd.Grouper(key="ds", freq="M")).sum().reset_index()

            if not forecast_df.empty:
                model = Prophet()
                model.fit(forecast_df)

                # Forecast next 2 months
                future = model.make_future_dataframe(periods=2, freq="M")
                forecast = model.predict(future)

                # Plot forecast
                fig1 = model.plot(forecast)
                st.pyplot(fig1)

                fig2 = model.plot_components(forecast)
                st.pyplot(fig2)

                # Show forecast table (last actual + 2 predicted months)
                st.subheader("📅 Forecast Table")
                forecast_table = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(3)
                forecast_table = forecast_table.rename(columns={
                    "ds": "Date",
                    "yhat": "Predicted",
                    "yhat_lower": "Lower Bound",
                    "yhat_upper": "Upper Bound"
                })
                st.dataframe(forecast_table)
            else:
                st.warning("⚠️ Not enough data for forecasting.")
        else:
            st.info("ℹ️ No valid date column detected for forecasting.")
    else:
        st.warning("⚠️ Please select at least one column.")

else:
    st.info("📂 Please upload a CSV file to continue.")