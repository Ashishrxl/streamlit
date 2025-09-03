import streamlit as st
import requests
import uuid

from rjsmin import jsmin  
hide_streamlit_style = """
    <style>
    #MainMenu {visibility: hidden;}      /* Hides the hamburger menu */
    header {visibility: hidden;}         /* Hides GitHub/Share buttons */
    footer {visibility: hidden;}         /* Hides "Made with Streamlit" footer */
    </style>
"""







disable_footer_link = """
    <style>
    /* Target the GitHub link in the footer specifically */
    footer a[href*="github.com"] {
        pointer-events: none !important;  /* Disable clicking */
        cursor: default !important;       /* Normal cursor */
        color: inherit !important;        /* Use normal text color */
        text-decoration: none !important; /* Remove underline */
    }
    </style>
"""
st.markdown(disable_footer_link, unsafe_allow_html=True)

# CSS to hide the "Made with Streamlit" footer
hide_footer_style = """
    <style>
    footer, footer:before {
        visibility: hidden;
    }
    div[data-testid="stDecoration"] {
        display: none;
    }
    div[data-testid="stStatusWidget"] {
        display: none;
    }
    div[data-testid="stToolbar"] {
        display: none;
    }
    </style>
"""



# Completely hide the Streamlit footer ("Made with Streamlit")
hide_footer = """
    <style>
    /* Target the main footer container */
    div[data-testid="stFooter"] {
        display: none;
    }
    </style>
"""



# Hide ONLY the GitHub repo link in footer
hide_github_footer = """
    <style>
    footer a {
        display: none !important;
    }
    </style>
"""



# Hide the "Created with/Hosted by Streamlit" footer
hide_streamlit_footer = """
    <style>
    div[data-testid="stFooter"] {
        display: none;
    }
    </style>
"""
st.markdown(hide_streamlit_footer, unsafe_allow_html=True)
st.markdown(hide_github_footer, unsafe_allow_html=True)
st.markdown(hide_footer, unsafe_allow_html=True)
st.markdown(hide_footer_style, unsafe_allow_html=True)
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

st.title("Hello, BharatGPT!")
st.write("This is your first Streamlit app using BharatGPT Mini.")

# User input
user_prompt = st.text_input(
    "Enter your prompt:", 
    "Explain generative AI in one sentence."
)

temperature = st.slider(
    "Model temperature:",
    min_value=0.0,
    max_value=1.0,
    value=0.7,
    step=0.01,
    help="Controls randomness: 0 = deterministic, 1 = very creative"
)

with st.spinner("AI is working..."):
    st.write("Hello")

# Server-side secret
app_id = st.secrets["corover"]["app_id"]

# Generate a temporary token for this session
session_token = str(uuid.uuid4())  # Unique token per user session

# Fetch widget JS server-side
def get_corover_widget(token):
    url = f"https://builder.corover.ai/params/widget/corovercb.lib.min.js?appId={app_id}&token={token}"
    response = requests.get(url)
    if response.status_code == 200:
        return response.text
    else:
        return f"// Failed to load widget: {response.status_code}"

widget_js = get_corover_widget(session_token)

# Minify / obfuscate JS
widget_js_min = jsmin(widget_js)

# Serve the widget in the app
st.components.v1.html(
    f"<script>{widget_js_min}</script>",
    height=800,
    scrolling=True
)


