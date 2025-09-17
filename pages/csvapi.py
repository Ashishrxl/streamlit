import streamlit as st
import pandas as pd
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain.agents import AgentType

# Set your Google API key as a Streamlit secret
GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]

# Initialize the Gemini model
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
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
    st.dataframe(df.head())

    # Create the pandas DataFrame agent with the recommended agent type for Gemini
    # It's crucial to also add allow_dangerous_code=True for the agent to work.
    agent = create_pandas_dataframe_agent(
        llm,
        df,
        verbose=True,
        agent_type=AgentType.TOOL_CALLING,  # Recommended fix
        allow_dangerous_code=True
    )

    # Chat input for user queries
    prompt = st.chat_input("Ask a question about your data...")
    if prompt:
        st.write(f"Thinking about: {prompt}")
        try:
            response = agent.invoke(prompt)
            st.write(response["output"])
        except Exception as e:
            st.error(f"An error occurred: {e}")
