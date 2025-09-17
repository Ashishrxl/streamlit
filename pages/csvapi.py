import streamlit as st
import pandas as pd
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent

# Set your Google API key as a Streamlit secret
# Go to "Manage app > Settings > Secrets" in Streamlit Cloud
# and add your GEMINI_API_KEY
GOOGLE_API_KEY = st.secrets["GEMINI_API_KEY"]

# Initialize the Gemini model with a low temperature for consistent output
llm = ChatGoogleGenerativeAI(
    model="gemini-pro",
    temperature=0.1,
    google_api_key=GOOGLE_API_KEY
)

st.title("CSV Data Explorer Agent 🤖")
st.write("Upload a CSV file and ask me anything about the data!")

# File uploader widget
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    # Read the CSV into a pandas DataFrame
    df = pd.read_csv(uploaded_file)
    st.success("File uploaded successfully!")
    st.dataframe(df.head()) # Display the first few rows

    # Create the pandas DataFrame agent
    agent = create_pandas_dataframe_agent(llm, df, verbose=True)

    # Chat input for user queries
    prompt = st.chat_input("Ask a question about your data...")
    if prompt:
        st.write(f"Thinking about: {prompt}")
        
        # Use a try-except block to handle potential errors from the agent
        try:
            # Invoke the agent with the user's prompt
            response = agent.invoke(prompt)
            # Display the final response from the agent
            st.write(response["output"])
            
        except Exception as e:
            st.error(f"An error occurred: {e}")
