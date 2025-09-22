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
import warnings
import logging
from datetime import datetime
import gc
import tempfile
import os

# Configure warnings and logging
warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Secure CSV Visualizer with Forecasting", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Security: Remove dangerous styling and add proper headers
st.markdown("""
<style>
    .main-header {
        padding: 1rem 0;
        border-bottom: 2px solid #e0e0e0;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeeba;
        color: #856404;
        padding: 1rem;
        border-radius: 4px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header"><h1>📊 Secure CSV Visualizer with Forecasting</h1></div>', unsafe_allow_html=True)

# Security and performance configuration
MAX_FILE_SIZE_MB = 100
CHUNK_SIZE = 10000
MAX_ROWS_DISPLAY = 1000

def validate_file_upload(uploaded_file):
    """Validate uploaded file for security and size constraints"""
    if uploaded_file is None:
        return False, "No file uploaded"
    
    if uploaded_file.size > MAX_FILE_SIZE_MB * 1024 * 1024:
        return False, f"File size exceeds {MAX_FILE_SIZE_MB}MB limit"
    
    if not uploaded_file.name.lower().endswith('.csv'):
        return False, "Only CSV files are allowed"
    
    return True, "File is valid"

def safe_read_csv(uploaded_file, nrows=None):
    """Safely read CSV file with error handling and memory management"""
    try:
        # Create temporary file for safer processing
        with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name
        
        # Read CSV with safety parameters
        df = pd.read_csv(
            tmp_file_path,
            low_memory=False,
            nrows=nrows,
            encoding='utf-8',
            on_bad_lines='skip'
        )
        
        # Clean up temporary file
        os.unlink(tmp_file_path)
        
        # Memory optimization
        for col in df.columns:
            if df[col].dtype == 'object':
                try:
                    df[col] = pd.to_numeric(df[col], errors='ignore')
                except:
                    pass
        
        return df, None
    
    except Exception as e:
        logger.error(f"Error reading CSV: {str(e)}")
        return None, f"Error reading CSV file: {str(e)}"

def find_column_case_insensitive(df: pd.DataFrame, target: str):
    """Find column by case-insensitive search"""
    for col in df.columns:
        if col.lower() == target.lower():
            return col
    return None

def convert_df_to_csv(df: pd.DataFrame) -> bytes:
    """Convert DataFrame to CSV bytes"""
    return df.to_csv(index=False).encode("utf-8")

def convert_df_to_excel(df: pd.DataFrame) -> bytes:
    """Convert DataFrame to Excel bytes"""
    try:
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
            df.to_excel(writer, index=False, sheet_name="Data")
        return buffer.getvalue()
    except Exception as e:
        logger.error(f"Error converting to Excel: {e}")
        return None

def export_plotly_fig(fig):
    """Export Plotly figure to PNG"""
    try:
        return pio.to_image(fig, format="png", width=1200, height=800)
    except Exception as e:
        logger.error(f"Error exporting Plotly figure: {e}")
        return None

def export_matplotlib_fig(fig):
    """Export Matplotlib figure to PNG"""
    try:
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", dpi=150)
        buf.seek(0)
        return buf.getvalue()
    except Exception as e:
        logger.error(f"Error exporting Matplotlib figure: {e}")
        return None

def safe_prophet_forecast(df, date_col, value_col, periods=6, freq='M'):
    """Safely perform Prophet forecasting with error handling"""
    try:
        # Prepare data for Prophet
        prophet_df = df[[date_col, value_col]].copy()
        prophet_df = prophet_df.rename(columns={date_col: "ds", value_col: "y"})
        prophet_df = prophet_df.dropna()
        
        if len(prophet_df) < 10:
            return None, "Insufficient data for forecasting (minimum 10 points required)"
        
        # Validate data types
        prophet_df['ds'] = pd.to_datetime(prophet_df['ds'], errors='coerce')
        prophet_df['y'] = pd.to_numeric(prophet_df['y'], errors='coerce')
        prophet_df = prophet_df.dropna()
        
        if len(prophet_df) < 10:
            return None, "Insufficient valid data after cleaning"
        
        # Initialize and fit Prophet model
        with st.spinner("Training forecasting model..."):
            model = Prophet(
                daily_seasonality=False,
                yearly_seasonality=True,
                weekly_seasonality=False,
                changepoint_prior_scale=0.05
            )
            model.fit(prophet_df)
            
            # Create future dataframe
            future = model.make_future_dataframe(periods=periods, freq=freq)
            forecast = model.predict(future)
            
        return forecast, None
    
    except Exception as e:
        logger.error(f"Error in Prophet forecasting: {e}")
        return None, f"Forecasting error: {str(e)}"

def create_safe_visualization(df, chart_type, x_col=None, y_col=None, hue_col=None):
    """Create visualizations with error handling"""
    try:
        fig = None
        
        if chart_type == "Scatter Plot" and x_col and y_col:
            fig = px.scatter(df, x=x_col, y=y_col, color=hue_col, 
                           title=f"Scatter Plot: {x_col} vs {y_col}")
        
        elif chart_type == "Line Chart" and x_col and y_col:
            fig = px.line(df, x=x_col, y=y_col, color=hue_col,
                         title=f"Line Chart: {x_col} vs {y_col}")
        
        elif chart_type == "Bar Chart" and x_col and y_col:
            fig = px.bar(df, x=x_col, y=y_col, color=hue_col,
                        title=f"Bar Chart: {x_col} vs {y_col}")
        
        elif chart_type == "Histogram" and x_col:
            fig = px.histogram(df, x=x_col, color=hue_col,
                             title=f"Histogram: {x_col}")
        
        elif chart_type == "Correlation Heatmap":
            numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if len(numerical_cols) >= 2:
                corr = df[numerical_cols].corr()
                fig = px.imshow(corr, text_auto=True, aspect='auto',
                               title="Correlation Heatmap",
                               color_continuous_scale='RdBu_r')
        
        if fig:
            fig.update_layout(height=500, showlegend=True)
            
        return fig, None
    
    except Exception as e:
        logger.error(f"Error creating visualization: {e}")
        return None, f"Visualization error: {str(e)}"

def process_uploaded_data(uploaded_df, is_alldata_format=False):
    """Process uploaded data and extract tables"""
    try:
        tables_dict = {"Original Data": uploaded_df}
        
        if is_alldata_format:
            # Extract specific tables for alldata format
            id_col = find_column_case_insensitive(uploaded_df, "ID")
            name_col = find_column_case_insensitive(uploaded_df, "Name")
            
            if id_col and name_col:
                party_df = uploaded_df[[id_col, name_col]].drop_duplicates().reset_index(drop=True)
                tables_dict["Party"] = party_df
            
            # Extract other tables similarly...
            bill_col = find_column_case_insensitive(uploaded_df, "Bill")
            partyid_col = find_column_case_insensitive(uploaded_df, "PartyId")
            date_col = find_column_case_insensitive(uploaded_df, "Date")
            amount_col = find_column_case_insensitive(uploaded_df, "Amount")
            
            if all([bill_col, partyid_col, date_col, amount_col]):
                bill_df = uploaded_df[[bill_col, partyid_col, date_col, amount_col]].drop_duplicates().reset_index(drop=True)
                tables_dict["Bill"] = bill_df
        
        return tables_dict
    
    except Exception as e:
        logger.error(f"Error processing data: {e}")
        return {"Original Data": uploaded_df}

def main():
    """Main application function"""
    
    # Sidebar configuration
    st.sidebar.header("⚙️ Application Settings")
    
    # Security notice
    st.sidebar.markdown("""
    <div class="warning-box">
    🔒 <strong>Security Notice:</strong><br>
    This application has been hardened for security. 
    AI chat features have been disabled to prevent code execution vulnerabilities.
    </div>
    """, unsafe_allow_html=True)
    
    # Configuration options
    max_rows_display = st.sidebar.slider("Max rows to display", 100, 2000, MAX_ROWS_DISPLAY)
    forecast_color = st.sidebar.color_picker("Forecast highlight color", "#FFA500")
    forecast_opacity = st.sidebar.slider("Forecast highlight opacity", 0.05, 1.0, 0.12)
    show_confidence = st.sidebar.checkbox("Show confidence intervals", True)
    
    # File upload section
    st.header("📁 File Upload")
    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type=["csv"],
        help=f"Maximum file size: {MAX_FILE_SIZE_MB}MB"
    )
    
    if uploaded_file is None:
        st.info("👆 Please upload a CSV file to begin analysis")
        st.markdown("""
        ### Features:
        - 📊 **Interactive Visualizations**: Multiple chart types with Plotly and Seaborn
        - 🔮 **Time Series Forecasting**: Prophet-based predictions with confidence intervals
        - 📈 **Data Processing**: Automatic table extraction and aggregation
        - 🔒 **Security Focused**: Hardened against common vulnerabilities
        - 📱 **Responsive Design**: Works on desktop and mobile devices
        """)
        return
    
    # Validate uploaded file
    is_valid, message = validate_file_upload(uploaded_file)
    if not is_valid:
        st.error(f"❌ {message}")
        return
    
    st.success("✅ File uploaded successfully!")
    
    # Read and process data
    with st.spinner("Processing file..."):
        df, error = safe_read_csv(uploaded_file)
    
    if error:
        st.error(f"❌ {error}")
        return
    
    if df is None or df.empty:
        st.error("❌ No data found in the uploaded file")
        return
    
    # Memory management
    if len(df) > max_rows_display * 10:
        st.warning(f"⚠️ Large dataset detected ({len(df)} rows). Using first {max_rows_display * 10} rows for analysis.")
        df = df.head(max_rows_display * 10)
    
    # Process data tables
    is_alldata_format = uploaded_file.name.lower() == "alldata.csv"
    tables_dict = process_uploaded_data(df, is_alldata_format)
    
    # Data preview section
    st.header("🗂️ Data Preview")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Rows", f"{len(df):,}")
    with col2:
        st.metric("Total Columns", len(df.columns))
    with col3:
        numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        st.metric("Numerical Columns", len(numerical_cols))
    with col4:
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        st.metric("Categorical Columns", len(categorical_cols))
    
    # Table selection and preview
    selected_table = st.selectbox("Select table to analyze", list(tables_dict.keys()))
    selected_df = tables_dict[selected_table].copy()
    
    # Display sample data
    with st.expander("📋 Data Sample", expanded=True):
        display_rows = min(20, len(selected_df))
        st.dataframe(selected_df.head(display_rows), use_container_width=True)
        
        # Download options
        col1, col2 = st.columns(2)
        with col1:
            csv_data = convert_df_to_csv(selected_df)
            st.download_button(
                "⬇️ Download CSV",
                data=csv_data,
                file_name=f"{selected_table.lower().replace(' ', '_')}.csv",
                mime="text/csv"
            )
        with col2:
            excel_data = convert_df_to_excel(selected_df)
            if excel_data:
                st.download_button(
                    "⬇️ Download Excel",
                    data=excel_data,
                    file_name=f"{selected_table.lower().replace(' ', '_')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
    
    # Visualization section
    st.header("📊 Data Visualization")
    
    with st.expander("Create Visualizations", expanded=False):
        if len(selected_df) == 0:
            st.warning("⚠️ No data available for visualization")
            return
        
        # Column selection
        all_columns = selected_df.columns.tolist()
        selected_columns = st.multiselect(
            "Select columns for analysis",
            all_columns,
            default=all_columns[:min(5, len(all_columns))]
        )
        
        if not selected_columns:
            st.warning("⚠️ Please select at least one column")
            return
        
        viz_df = selected_df[selected_columns].copy()
        
        # Chart configuration
        chart_options = [
            "Scatter Plot", "Line Chart", "Bar Chart", "Histogram", 
            "Correlation Heatmap"
        ]
        
        chart_type = st.selectbox("Select chart type", chart_options)
        
        # Dynamic column selection based on chart type
        x_col = y_col = hue_col = None
        
        if chart_type in ["Scatter Plot", "Line Chart", "Bar Chart"]:
            col1, col2, col3 = st.columns(3)
            with col1:
                x_col = st.selectbox("X-axis", viz_df.columns)
            with col2:
                y_options = [col for col in viz_df.columns if col != x_col]
                y_col = st.selectbox("Y-axis", y_options) if y_options else None
            with col3:
                hue_options = ["None"] + [col for col in viz_df.columns if col not in [x_col, y_col]]
                hue_selection = st.selectbox("Color/Hue", hue_options)
                hue_col = hue_selection if hue_selection != "None" else None
        
        elif chart_type == "Histogram":
            x_col = st.selectbox("Column to analyze", viz_df.columns)
            hue_options = ["None"] + [col for col in viz_df.columns if col != x_col]
            hue_selection = st.selectbox("Color/Hue", hue_options)
            hue_col = hue_selection if hue_selection != "None" else None
        
        # Create visualization
        if st.button("🎨 Generate Chart"):
            fig, error = create_safe_visualization(viz_df, chart_type, x_col, y_col, hue_col)
            
            if error:
                st.error(f"❌ {error}")
            elif fig:
                st.plotly_chart(fig, use_container_width=True)
                
                # Export option
                png_data = export_plotly_fig(fig)
                if png_data:
                    st.download_button(
                        "⬇️ Download Chart (PNG)",
                        data=png_data,
                        file_name=f"{chart_type.lower().replace(' ', '_')}.png",
                        mime="image/png"
                    )
    
    # Forecasting section
    st.header("🔮 Time Series Forecasting")
    
    with st.expander("Prophet Forecasting", expanded=False):
        # Find date and numeric columns
        date_columns = []
        for col in selected_df.columns:
            if 'date' in col.lower() or selected_df[col].dtype == 'datetime64[ns]':
                date_columns.append(col)
        
        numeric_columns = selected_df.select_dtypes(include=[np.number]).columns.tolist()
        
        if not date_columns:
            st.info("ℹ️ No date columns found. Trying to detect date columns...")
            # Try to convert potential date columns
            for col in selected_df.columns:
                try:
                    pd.to_datetime(selected_df[col].head(), errors='raise')
                    date_columns.append(col)
                except:
                    continue
        
        if not date_columns or not numeric_columns:
            st.warning("⚠️ Forecasting requires both date and numeric columns")
        else:
            col1, col2 = st.columns(2)
            with col1:
                date_col = st.selectbox("Select date column", date_columns)
            with col2:
                value_col = st.selectbox("Select value column", numeric_columns)
            
            col1, col2 = st.columns(2)
            with col1:
                forecast_periods = st.slider("Forecast periods", 1, 24, 6)
            with col2:
                frequency = st.selectbox("Frequency", ["M", "D", "W"], index=0)
            
            if st.button("🚀 Generate Forecast"):
                # Prepare data for forecasting
                forecast_df = selected_df[[date_col, value_col]].copy()
                forecast_df[date_col] = pd.to_datetime(forecast_df[date_col], errors='coerce')
                forecast_df = forecast_df.dropna()
                
                if len(forecast_df) < 10:
                    st.error("❌ Insufficient data for forecasting (minimum 10 points required)")
                else:
                    # Perform forecasting
                    forecast_result, error = safe_prophet_forecast(
                        forecast_df, date_col, value_col, forecast_periods, frequency
                    )
                    
                    if error:
                        st.error(f"❌ {error}")
                    elif forecast_result is not None:
                        # Create forecast visualization
                        fig = px.line(
                            title=f"Forecast: {value_col} over {date_col}"
                        )
                        
                        # Historical data
                        fig.add_scatter(
                            x=forecast_df[date_col],
                            y=forecast_df[value_col],
                            mode='lines+markers',
                            name='Historical',
                            line=dict(color='blue')
                        )
                        
                        # Forecast data
                        future_data = forecast_result[forecast_result['ds'] > forecast_df[date_col].max()]
                        fig.add_scatter(
                            x=future_data['ds'],
                            y=future_data['yhat'],
                            mode='lines+markers',
                            name='Forecast',
                            line=dict(color='orange', dash='dash')
                        )
                        
                        if show_confidence:
                            # Confidence intervals
                            fig.add_scatter(
                                x=future_data['ds'],
                                y=future_data['yhat_upper'],
                                mode='lines',
                                line=dict(width=0),
                                showlegend=False,
                                name='Upper Bound'
                            )
                            
                            fig.add_scatter(
                                x=future_data['ds'],
                                y=future_data['yhat_lower'],
                                mode='lines',
                                line=dict(width=0),
                                fill='tonexty',
                                fillcolor=f'rgba(255, 165, 0, {forecast_opacity})',
                                showlegend=True,
                                name='Confidence Interval'
                            )
                        
                        fig.update_layout(height=500)
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Forecast table
                        st.subheader("📅 Forecast Results")
                        forecast_table = future_data[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].copy()
                        forecast_table.columns = ['Date', 'Predicted', 'Lower Bound', 'Upper Bound']
                        forecast_table = forecast_table.round(2)
                        
                        st.dataframe(forecast_table, use_container_width=True)
                        
                        # Download forecast results
                        csv_forecast = convert_df_to_csv(forecast_table)
                        st.download_button(
                            "⬇️ Download Forecast Results",
                            data=csv_forecast,
                            file_name="forecast_results.csv",
                            mime="text/csv"
                        )
    
    # Information section
    st.header("ℹ️ Application Information")
    
    with st.expander("About this Application"):
        st.markdown("""
        ### Security Features:
        - ✅ File validation and size limits
        - ✅ Safe data processing with error handling
        - ✅ Memory management for large files
        - ✅ Removed dangerous code execution capabilities
        
        ### Capabilities:
        - 📊 Interactive data visualization
        - 🔮 Time series forecasting with Prophet
        - 📈 Multiple chart types and customization
        - 💾 Data export in CSV and Excel formats
        - 📱 Responsive design for all devices
        
        ### Technical Stack:
        - **Frontend**: Streamlit
        - **Data Processing**: Pandas, NumPy
        - **Visualization**: Plotly, Seaborn, Matplotlib
        - **Forecasting**: Facebook Prophet
        - **Security**: Input validation, memory management
        """)
    
    # Performance cleanup
    gc.collect()

if __name__ == "__main__":
    main()
