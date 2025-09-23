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
import google.generativeai as genai
import json
from coree import run_app_logic

# New imports for LangChain agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain.agents import AgentType

st.set_page_config(page_title="CSV Visualizer with Forecasting (Interactive)", layout="wide")
st.title("📊 CSV Visualizer with Forecasting (Interactive)")

# Use Streamlit secrets for API key
try:
    genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
    gemini_model = genai.GenerativeModel('gemini-1.5-flash') # Renamed for clarity

    # Initialize the LangChain Gemini model            
    llm = ChatGoogleGenerativeAI(            
        model="gemini-1.5-flash",            
        temperature=0.1,            
        google_api_key=st.secrets["GOOGLE_API_KEY"]            
    )

except Exception as e:
    st.error(f"Error configuring Gemini API: {e}. Please ensure GOOGLE_API_KEY is set in your Streamlit secrets.")
    st.stop()

hide_streamlit_style = """
<style>            
#MainMenu, footer {visibility: hidden;} footer {display: none !important;} header {display: none !important;} #MainMenu {display: none !important;} [data-testid="stToolbar"] { display: none !important; } .st-emotion-cache-1xw8zd0 {display: none !important;} [aria-label="View app source"] {display: none !important;} a[href^="https://github.com"] {display: none !important;} [data-testid="stDecoration"] {display: none !important;} [data-testid="stStatusWidget"] {display: none !important;} button[title="Menu"] {display: none !important;}            
</style>            """
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
        # Fixed: Removed deprecated engine parameter
        return pio.to_image(fig, format="png")
    except Exception:
        return None

def export_matplotlib_fig(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    return buf.getvalue()

def toggle_state(key):
    st.session_state[key] = not st.session_state[key]


          
# --- Main App Logic ---

st.sidebar.header("⚙️ Settings")
forecast_color = st.sidebar.color_picker("Forecast highlight color", "#FFA500")
forecast_opacity = st.sidebar.slider("Forecast highlight opacity", 0.05, 1.0, 0.12, step=0.01)
show_confidence = st.sidebar.checkbox("Show confidence interval (upper/lower bounds)", True)

uploaded_file = st.file_uploader("Upload your CSV file !", type=["csv"])
if uploaded_file is None:
    st.info("Upload a CSV to start. The app will derive tables and let you visualize/forecast.")
    st.stop()

file_name = uploaded_file.name

if file_name.lower() == "alldata.csv":
    try:
        uploaded_df = pd.read_csv(uploaded_file, low_memory=False)
        st.success("✅ File uploaded successfully!")
        run_app_logic(uploaded_df, is_alldata=True)
    except Exception as e:
        st.error(f"❌ Error reading CSV: {e}")
        st.stop()
else:
    st.warning("⚠️ Please confirm its structure.")
    st.subheader("📋 Confirm File Structure")

    header_option = st.radio("Does your CSV file have a header row?", ["Yes", "No"])            
    if header_option == "Yes":            
        try:            
            uploaded_df = pd.read_csv(uploaded_file, low_memory=False)            
            st.success("✅ File loaded with header successfully!")            
            st.info("Now, please confirm the column names for analysis.")            
            col_confirm = st.radio("Are the column names correct?", ["Yes", "No, I want to rename them"])            

            if col_confirm == "Yes":            
                st.success("Column names confirmed. Proceeding with visualization and forecasting.")            
                run_app_logic(uploaded_df, is_alldata=False)            
            elif col_confirm == "No, I want to rename them":            
                st.info("Please provide the new column names.")            
                new_cols_dict = {}            
                original_cols = uploaded_df.columns.tolist()            
                # Create form for column renaming            
                with st.form("column_rename_form"):            
                    st.write("### Rename Columns")            
                    for i, col in enumerate(original_cols):            
                        new_name = st.text_input(f"Rename '{col}' to:", value=col, key=f"rename_{i}")            
                        new_cols_dict[col] = new_name if new_name.strip() else col            
                    # Submit button for the form            
                    submitted = st.form_submit_button("✅ Apply Column Renames", type="primary")            
                    if submitted:            
                        try:            
                            uploaded_df = uploaded_df.rename(columns=new_cols_dict)            
                            st.success("✅ Columns renamed successfully!")            
                            st.info("**Updated column names:**")            
                            st.write(list(uploaded_df.columns))            
                            run_app_logic(uploaded_df, is_alldata=False)            
                        except Exception as rename_error:            
                            st.error(f"❌ Error renaming columns: {rename_error}")            
        except Exception as e:            
            st.error(f"❌ Error reading CSV: {e}")            
            st.stop()            
    else:            
        try:            
            uploaded_df = pd.read_csv(uploaded_file, header=None, low_memory=False)            
            st.success("✅ File loaded without header successfully!")            
            st.warning("⚠️ Auto-generating column names...")            
            # Auto-generate column names            
            num_cols = len(uploaded_df.columns)            
            default_cols = [f"Column_{i+1}" for i in range(num_cols)]            
            uploaded_df.columns = default_cols            
            st.info(f"📊 Generated {num_cols} column names: {', '.join(default_cols[:5])}{'...' if num_cols > 5 else ''}")            
            # Option to customize column names            
            if st.checkbox("🏷️ Customize column names"):            
                with st.form("auto_column_rename"):            
                    st.write("### Customize Auto-Generated Column Names")            
                    custom_cols = {}            
                    for i, col in enumerate(default_cols):            
                        custom_name = st.text_input(f"Column {i+1} name:", value=col, key=f"custom_{i}")            
                        custom_cols[col] = custom_name if custom_name.strip() else col            
                    if st.form_submit_button("✅ Apply Custom Names"):            
                        uploaded_df = uploaded_df.rename(columns=custom_cols)            
                        st.success("✅ Column names updated!")            
            run_app_logic(uploaded_df, is_alldata=False)            
        except Exception as e:            
            st.error(f"❌ Error reading CSV without header: {e}")            
            st.info("💡 Please check if your file is a valid CSV format.")
