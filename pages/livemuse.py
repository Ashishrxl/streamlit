import streamlit as st
import requests
import base64
import io
import numpy as np
import wave
from scipy.io import wavfile
from streamlit.components.v1 import html
from streamlit_webrtc import webrtc_streamer, WebRtcMode, AudioProcessorBase
import av

# --- Hide Streamlit branding ---
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
    page_title="🎙️ LiveMuse",
    page_icon="🌐",
    layout="centered",
    initial_sidebar_state="expanded",
)

# --- Hide top menu and toolbar ---
hide_streamlit_style = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
[data-testid="stStatusWidget"] {display: none;}
[data-testid="stToolbar"] {display: none;}
a[href^="https://github.com"] {display: none !important;}
a[href^="https://streamlit.io"] {display: none !important;}
header > div:nth-child(2) {display: none;}
</style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# --- Title ---
st.title("🎵 LiveMuse – Real-time AI Music Co-Creation")
st.write("Hum, beatbox, or record a clip — Gemini will turn your idea into music 🎧")

# --- API Key selection ---
api_keys = {
    "Key 1": st.secrets["KEY_1"],
    "Key 2": st.secrets["KEY_2"], "Key 3": st.secrets["KEY_3"], "Key 4": st.secrets["KEY_4"], "Key 5": st.secrets["KEY_5"], "Key 6": st.secrets["KEY_6"], "Key 7": st.secrets["KEY_7"], "Key 8": st.secrets["KEY_8"], "Key 9": st.secrets["KEY_9"], "Key 10": st.secrets["KEY_10"], "Key 11": st.secrets["KEY_11"]
}
selected_key_name = st.selectbox("Select API Key", list(api_keys.keys()))
api_key = api_keys[selected_key_name]

# --- Sidebar settings ---
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
instrument = st.sidebar.selectbox(
    "Target Style", ["Piano", "Lo-fi Beat", "Synth Pad", "Guitar", "Ambient"]
)

# --- WebRTC Audio Recorder ---
st.header("1️⃣ Record your seed audio")

# Buffer to hold recorded frames
recorded_frames = []

# Define Audio Processor
class AudioProcessor(AudioProcessorBase):
    def recv_audio_frame(self, frame: av.AudioFrame) -> av.AudioFrame:
        recorded_frames.append(frame)
        return frame

# Create WebRTC streamer
webrtc_ctx = webrtc_streamer(
    key="audio-recorder",
    mode=WebRtcMode.SENDONLY,
    audio_receiver_size=256,
    media_stream_constraints={"audio": True, "video": False},
    audio_processor_factory=AudioProcessor,
)

seed_audio = None

if st.button("⏹ Stop & Save Recording"):
    if len(recorded_frames) > 0:
        st.info("Processing your recording...")
        audio_data = b""
        for f in recorded_frames:
            sound = f.to_ndarray().astype(np.int16).tobytes()
            audio_data += sound

        # Convert to WAV
        sample_rate = recorded_frames[0].sample_rate
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(audio_data)

        seed_audio = buffer.getvalue()
        st.audio(seed_audio, format="audio/wav")
        st.success("✅ Recording saved successfully!")
    else:
        st.warning("No audio recorded yet. Please record first.")

# --- Generate Music ---
st.header("2️⃣ Generate AI Music")

if st.button("🎶 Generate with Gemini") and seed_audio:
    with st.spinner("Calling Gemini model..."):
        api_key = api_key
        if not api_key:
            st.error("Missing GOOGLE_API_KEY in Streamlit secrets.")
            st.stop()

        # Encode audio
        audio_b64 = base64.b64encode(seed_audio).decode("utf-8")

        # API endpoint
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_choice}:generateContent"

        # Prompt
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
                                "data": audio_b64,
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

        # Call Gemini API
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=120)
            response.raise_for_status()
            data = response.json()

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

elif st.button("🎶 Generate with Google") and not seed_audio:
    st.warning("Please record audio first!")

st.markdown("---")
st.caption("Powered by Gemini 2.5 Flash Native Audio — via Google AI Studio API key")