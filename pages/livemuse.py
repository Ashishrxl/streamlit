import streamlit as st
import requests
import base64
import io
import numpy as np
from scipy.io import wavfile
from streamlit.components.v1 import html
from streamlit_audio_recorder import audio_recorder  # ✅ updated import

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

st.set_page_config(
    page_title="My App",
    page_icon="🌐",
    initial_sidebar_state="expanded"
)


# --- CSS: Hide all unwanted items but KEEP sidebar toggle ---
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


st.set_page_config(page_title="🎙️ LiveMuse", layout="centered")

st.title("🎵 LiveMuse – Real-time AI Music Co-Creation")
st.write("Hum, beatbox, or upload a clip — Gemini will turn your idea into music 🎧")

# --- Sidebar ---

model_choice = st.selectbox(
    "Gemini Audio Model",
    [
        "models/gemini-2.5-flash-native-audio-preview-09-2025",
        "models/gemini-2.5-flash-native-audio-latest",
        "models/gemini-2.5-flash-preview-native-audio-dialog",
    ],
)
tempo = st.slider("Tempo (BPM)", 60, 160, 100)
duration = st.slider("Desired output length (seconds)", 5, 30, 15)
instrument = st.sidebar.selectbox("Target Style", ["Piano", "Lo-fi Beat", "Synth Pad", "Guitar", "Ambient"])


# --- Audio input ---
st.header("1️⃣ Record your seed audio")

recorded_audio_enhanced = audio_recorder(
    text="Click to record",
    recording_color="#e8b62c",
    neutral_color="#6aa36f",
    icon_name="microphone",
    icon_size="2x",
    key="enhanced_recorder"
)

if recorded_audio_enhanced:
    st.audio(recorded_audio_enhanced, format="audio/wav")
    seed_audio = recorded_audio_enhanced
else:
    st.info("Press the record button and hum or beatbox for 5–10 seconds 🎙️")


# --- Generate ---
st.header("2️⃣ Generate AI Music")

if st.button("🎶 Generate with Gemini") and recorded_audio_enhanced:
    st.spinner("Calling Gemini model...")

    api_key = st.secrets.get("GOOGLE_API_KEY")
    if not api_key:
        st.error("Missing GOOGLE_API_KEY in Streamlit secrets.")
        st.stop()

    # Encode audio to base64
    audio_b64 = base64.b64encode(seed_audio).decode("utf-8")

    # Gemini REST endpoint
    url = "https://generativelanguage.googleapis.com/v1beta/models/" + model_choice + ":generateContent"

    # Create the request payload
    prompt = (
        f"Transform this vocal or beat idea into a short {instrument} loop "
        f"at {tempo} BPM, around {duration} seconds long. "
        "Make it sound musical and natural."
    )

    payload = {
        "contents": [
            {
                "parts": [
                    {"text": prompt},
                    {
                        "inline_data": {
                            "mime_type": "audio/wav",
                            "data": audio_b64
                        }
                    },
                ]
            }
        ]
    }

    headers = {
        "Content-Type": "application/json",
        "x-goog-api-key": api_key,
    }

    # Send request
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=120)
        response.raise_for_status()
        data = response.json()

        # Parse response — Gemini audio models may return audio in base64 or text
        generated_audio = None
        if "candidates" in data:
            for c in data["candidates"]:
                parts = c.get("content", {}).get("parts", [])
                for p in parts:
                    if "inline_data" in p and p["inline_data"].get("mime_type", "").startswith("audio"):
                        generated_audio = base64.b64decode(p["inline_data"]["data"])
                        break

        if generated_audio:
            st.success("✅ Music generated successfully!")
            st.audio(generated_audio)

            b64 = base64.b64encode(generated_audio).decode()
            st.markdown(
                f'<a href="data:audio/wav;base64,{b64}" download="livemuse_output.wav">📥 Download AI Music</a>',
                unsafe_allow_html=True,
            )
        else:
            st.error("No audio data returned from Gemini.")
            st.json(data)

    except requests.exceptions.RequestException as e:
        st.error(f"API request failed: {e}")

elif st.button("🎶 Generate with Google") and not recorded_audio_enhanced:
    st.warning("Please record audio first!")

st.markdown("---")
st.caption("Powered by Gemini 2.5 Flash Native Audio — via Google AI Studio API key")