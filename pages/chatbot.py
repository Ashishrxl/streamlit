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
import re

st.set_page_config(page_title="CSV Visualizer with Forecasting (Interactive)", layout="wide")
st.title("📊 CSV Visualizer with Forecasting (Interactive)")

# Use Streamlit secrets for API key
try:
    genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
    gemini_model = genai.GenerativeModel('gemini-1.5-flash')
except Exception as e:
    st.error(f"Error configuring Gemini API: {e}. Please ensure GOOGLE_API_KEY is set in your Streamlit secrets.")
    st.stop()


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

# === Chat Execution Helper ===
def execute_generated_code(generated_code, df):
    """Safely execute AI-generated code with support for DataFrames, Matplotlib, and Plotly."""
    exec_globals = {'df': df, 'result': None, 'pd': pd, 'np': np, 'plt': plt, 'px': px}

    try:
        exec(generated_code, exec_globals)
        analysis_result = exec_globals.get('result', None)

        # Automatically render plots if generated
        if isinstance(analysis_result, (pd.DataFrame, pd.Series)):
            st.dataframe(analysis_result)
        elif isinstance(analysis_result, (plt.Figure,)):
            st.pyplot(analysis_result)
        elif isinstance(analysis_result, (dict, list)):
            st.json(analysis_result)
        elif analysis_result is not None:
            st.write(analysis_result)

        # Check if a matplotlib figure was created
        if plt.get_fignums():
            st.pyplot(plt.gcf())
            plt.close("all")

        # Check if a Plotly figure was created
        if isinstance(analysis_result, (px.Figure,)):
            st.plotly_chart(analysis_result, use_container_width=True)

        return analysis_result

    except Exception as e:
        error_message = f"⚠️ Error while executing generated code: {e}"
        st.error(error_message)
        return error_message

def run_app_logic(uploaded_df, is_alldata):
    # ... KEEP YOUR EXISTING LOGIC (tables, visualization, forecasting) ...
    # I won’t repeat unchanged sections for brevity

    # --- Chat with CSV Section ---
    st.markdown("---")
    with st.expander("🤖 Chat with your CSV", expanded=False):
        st.subheader("📌 Select Table for Chat")
        available_tables_chat = {k: v for k, v in tables_dict.items() if not v.empty}
        if not available_tables_chat:
            st.warning("⚠️ No usable tables could be derived from the uploaded CSV.")
            st.stop()

        selected_table_name_chat = st.selectbox("Select one table to chat with", list(available_tables_chat.keys()), key="chat_table_select")
        df = available_tables_chat[selected_table_name_chat].copy()

        # Display the preview of the selected table
        st.write(f"### Preview of '{selected_table_name_chat}'")
        st.dataframe(df.head(10))

        # Initialize chat history for this section
        if "chat_messages" not in st.session_state:
            st.session_state.chat_messages = [{"role": "assistant", "content": "Hello! I can help you analyze this data. Ask a question like, 'What's the average of the Amount column?' or 'Show me the top 5 bills by total amount.' I will write and run Python code to get the answer."}]

        # Display chat messages from history on app rerun
        for message in st.session_state.chat_messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Accept user input
        if prompt := st.chat_input("Ask me about the data..."):
            # Add user message to chat history
            st.session_state.chat_messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            # Prepare the prompt for Gemini to generate code
            df_info = f"DataFrame columns:\n{df.columns.tolist()}\n\nDataFrame dtypes:\n{df.dtypes.to_string()}\n\nFirst 5 rows:\n{df.head().to_string()}"

            code_prompt = f"""
You are a Python data analyst assistant. Your task is to write a single, clean block of Python code to answer a user's question about a pandas DataFrame named `df`.

The user's DataFrame has the following structure:
{df_info}

The user's question is: "{prompt}"

Write a Python script that completes the following steps:
1. Perform the necessary data analysis on the `df` DataFrame.
2. The final result should be stored in a variable named `result`.
3. Do not include any print statements. The system will automatically display the content of the `result` variable.
4. If you generate a Matplotlib or Plotly figure, store it in `result`.
5. **DO NOT** use any special characters or markdown formatting (e.g., ```python) in your response. Just provide the code.
"""

            with st.chat_message("assistant"):
                with st.spinner("Analyzing data and generating code..."):
                    try:
                        # Generate the Python code from the AI
                        code_response = gemini_model.generate_content(code_prompt)
                        generated_code = code_response.text.strip()

                        # Show generated code
                        st.text(f"Generated Code:\n{generated_code}")
                        st.subheader("Result:")

                        # Execute and render results
                        analysis_result = execute_generated_code(generated_code, df)

                        # Store in chat history
                        st.session_state.chat_messages.append({
                            "role": "assistant",
                            "content": f"Generated Code:\n```python\n{generated_code}\n```\n\nResult:\n{str(analysis_result)}"
                        })

                    except Exception as e:
                        error_message = f"An error occurred while generating the code: {e}"
                        st.error(error_message)
                        st.session_state.chat_messages.append({"role": "assistant", "content": error_message})

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

                for col in original_cols:
                    new_name = st.text_input(f"Rename column '{col}':", value=col)
                    new_cols_dict[col] = new_name

                if st.button("Apply Renaming and Analyze"):
                    try:
                        renamed_df = uploaded_df.rename(columns=new_cols_dict)
                        st.success("Columns renamed successfully! Analyzing the new data structure.")
                        run_app_logic(renamed_df, is_alldata=False)
                    except Exception as e:
                        st.error(f"❌ Failed to rename columns: {e}")
        except Exception as e:
            st.error(f"❌ Error reading CSV with header: {e}")
            st.stop()
    elif header_option == "No":
        try:
            uploaded_df = pd.read_csv(uploaded_file, header=None, low_memory=False)
            st.success("✅ File loaded without header successfully!")
            st.info("Please rename the generic columns to meaningful names.")

            new_cols_dict = {}
            original_cols = uploaded_df.columns.tolist()

            for col in original_cols:
                new_name = st.text_input(f"Rename generic column '{col}' (e.g., Column 0):", value="")
                new_cols_dict[col] = new_name

            if st.button("Apply Renaming and Analyze"):
                try:
                    renamed_df = uploaded_df.rename(columns=new_cols_dict)
                    st.success("Columns renamed successfully! Analyzing the new data structure.")
                    run_app_logic(renamed_df, is_alldata=False)
                except Exception as e:
                    st.error(f"❌ Failed to rename columns: {e}")
        except Exception as e:
            st.error(f"❌ Error reading CSV without header: {e}")
            st.stop()