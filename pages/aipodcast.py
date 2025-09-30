# --- Replace your google imports with these ---
from google import genai
from google.genai import types

import base64
import wave
import streamlit as st

from streamlit.components.v1 import html


html(
  """
  <script>
  try {
    const sel = window.top.document.querySelectorAll('[href*="streamlit.io"], [href*="streamlit.app"]');
    sel.forEach(e => e.style.display='none');
  } catch(e) { console.warn('parent DOM not reachable', e); }
  </script>
  """,
  height=0
)

disable_footer_click = """
    <style>
    footer {pointer-events: none;}
    </style>
"""
st.markdown(disable_footer_click, unsafe_allow_html=True)

hide_streamlit_style = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
[data-testid="stStatusWidget"] {display: none;}
[data-testid="stToolbar"] {display: none;}
a[href^="https://github.com"] {display: none !important;}
a[href^="https://streamlit.io"] {display: none !important;}

/* The following specifically targets and hides all child elements of the header's right side,
   while preserving the header itself and, by extension, the sidebar toggle button. */
header > div:nth-child(2) {
    display: none;
}
</style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# Create client (uses API key from streamlit secrets)
client = genai.Client(api_key=st.secrets["GOOGLE_API_KEY"])
# ---------------------------------------------------------------------

def generate_script(topic):
    prompt = f"""
    Write a friendly and engaging podcast script about "{topic}".
    Include:
    - A short intro
    - 3 key talking points
    - A closing statement
    Keep it conversational and natural.
    """
    # Use a text model for generation
    resp = client.models.generate_content(
        model="gemini-2.5-pro",
        contents=prompt
    )
    # The SDK exposes .text for text responses
    return resp.text

def save_wave(filename, pcm_data, channels=1, rate=24000, sample_width=2):
    with wave.open(filename, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(sample_width)
        wf.setframerate(rate)
        wf.writeframes(pcm_data)

def generate_audio(script_text, voice_name="Kore", language="English"):
    # style prompt is OK to include in the text you send to the TTS model
    if language.lower() == "hindi":
        style_prompt = "Speak this in a warm and expressive Hindi accent."
    elif language.lower() == "bhojpuri":
        style_prompt = "Speak this in a friendly Bhojpuri tone, like a local storyteller."
    else:
        style_prompt = "Speak this in a natural and friendly tone."

    contents = f"{style_prompt}\n\n{script_text}"

    # Use the TTS model and the GenerateContentConfig shape expected by the SDK
    response = client.models.generate_content(
        model="gemini-2.5-flash-preview-tts",
        contents=contents,
        config=types.GenerateContentConfig(
            response_modalities=["AUDIO"],
            speech_config=types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(
                        voice_name=voice_name
                    )
                )
            )
        )
    )

    # Extract PCM audio. The SDK example uses this path.
    pcm_data = response.candidates[0].content.parts[0].inline_data.data

    # Defensive: if the SDK returned a base64 string, decode it
    if isinstance(pcm_data, str):
        pcm_data = base64.b64decode(pcm_data)

    filename = "podcast.wav"
    save_wave(filename, pcm_data)
    return filename