import packages

from dotenv import load_dotenv 
import streamlit as st 
import bharatgpt   # <-- Replaced OpenAI with BharatGPT package

load environment variables from .env file

load_dotenv()

Initialize BharatGPT client

client = bharatgpt.Client()

@st.cache_data def get_response(user_prompt, temperature): response = client.responses.create( model="bharatgpt-mini",  # Use BharatGPT Mini model input=[ {"role": "user", "content": user_prompt}  # Prompt ], temperature=temperature,  # A bit of creativity max_output_tokens=100  # Limit response length ) return response

st.title("Hello, BharatGPT!") st.write("This is your first Streamlit app using BharatGPT Mini.")

Add a text input box for the user prompt

user_prompt = st.text_input("Enter your prompt:", "Explain generative AI in one sentence.")

Add a slider for temperature

temperature = st.slider( "Model temperature:", min_value=0.0, max_value=1.0, value=0.7, step=0.01, help="Controls randomness: 0 = deterministic, 1 = very creative" )

with st.spinner("AI is working..."): response = get_response(user_prompt, temperature) # print the response from BharatGPT st.write(response.output[0].content[0].text)

Inject Corover widget JavaScript import packages

from dotenv import load_dotenv import streamlit as st import bharatgpt   # <-- Replaced OpenAI with BharatGPT package

load environment variables from .env file

load_dotenv()

Initialize BharatGPT client

client = bharatgpt.Client()

@st.cache_data def get_response(user_prompt, temperature): response = client.responses.create( model="bharatgpt-mini",  # Use BharatGPT Mini model input=[ {"role": "user", "content": user_prompt}  # Prompt ], temperature=temperature,  # A bit of creativity max_output_tokens=100  # Limit response length ) return response

st.title("Hello, BharatGPT!") st.write("This is your first Streamlit app using BharatGPT Mini.")

Add a text input box for the user prompt

user_prompt = st.text_input("Enter your prompt:", "Explain generative AI in one sentence.")

Add a slider for temperature

temperature = st.slider( "Model temperature:", min_value=0.0, max_value=1.0, value=0.7, step=0.01, help="Controls randomness: 0 = deterministic, 1 = very creative" )

with st.spinner("AI is working..."): response = get_response(user_prompt, temperature) # print the response from BharatGPT st.write(response.output[0].content[0].text)

Inject Corover widget JavaScript into Streamlit

tool_script = """

<script type="text/javascript">
    var s = document.createElement("script");
    s.src = "https://builder.corover.ai/params/widget/corovercb.lib.min.js?appId=d977ea27-874f-411c-a08e-cc352dcdf0f2";
    s.type = "text/javascript";
    document.getElementsByTagName("body")[0].appendChild(s);
</script>"""

Display chatbot widget

st.components.v1.html(tool_script, height=800, scrolling=True)


