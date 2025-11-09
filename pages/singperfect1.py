import streamlit as st
import websocket
import json
import base64
import threading
import queue
import sounddevice as sd
import tempfile
import time
import os

# ========================================
# CONFIG
# ========================================
st.set_page_config(page_title="🎤 Live AI Vocal Coach", layout="wide")

st.markdown("""
<style>
footer {visibility:hidden;}
#MainMenu {visibility:hidden;}
</style>
""", unsafe_allow_html=True)

st.title("🎵 Live AI Vocal Coach (Gemini Realtime)")

# ========================================
# API KEY
# ========================================
if "GOOGLE_API_KEY_1" not in st.secrets:
    st.error("❌ Missing GOOGLE_API_KEY in Streamlit secrets.")
    st.stop()

API_KEY = st.secrets["GOOGLE_API_KEY_1"]

# ========================================
# Correct Gemini Realtime WebSocket endpoint
# ========================================
REALTIME_URL = (
    "wss://generativelanguage.googleapis.com/ws/"
    "google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent"
    f"?key={API_KEY}"
)

# ========================================
# Globals
# ========================================
q = queue.Queue()
stop_flag = threading.Event()

# ========================================
# Audio Stream to Gemini
# ========================================
def stream_microphone_audio(ws):
    """Capture microphone input and send to Gemini as base64 audio."""
    def callback(indata, frames, time, status):
        if stop_flag.is_set():
            raise sd.CallbackStop()
        audio_b64 = base64.b64encode(indata).decode("utf-8")
        msg = {"input_audio_buffer": {"data": audio_b64}}
        ws.send(json.dumps(msg))

    with sd.RawInputStream(
        samplerate=16000, blocksize=1024, dtype="int16", channels=1, callback=callback
    ):
        while not stop_flag.is_set():
            time.sleep(0.05)

    # tell model input is complete
    ws.send(json.dumps({"input_audio_buffer": {"complete": True}}))


# ========================================
# Handle Gemini Responses
# ========================================
def receive_from_gemini(ws):
    """Continuously receive responses from Gemini."""
    while not stop_flag.is_set():
        try:
            msg = ws.recv()
        except websocket.WebSocketConnectionClosedException:
            break
        if not msg:
            continue

        data = json.loads(msg)
        if data.get("response") and data["response"].get("output"):
            outputs = data["response"]["output"]

            # Process text response
            for part in outputs:
                if "text" in part:
                    st.write("🗣️ " + part["text"])

                # Process audio response
                if "data" in part:
                    audio_bytes = base64.b64decode(part["data"])
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
                        f.write(audio_bytes)
                        audio_path = f.name
                    st.audio(audio_path, format="audio/wav")


# ========================================
# Main Live Coaching Logic
# ========================================
def start_live_coaching():
    """Start WebSocket connection and stream audio to Gemini."""
    st.info("Connecting to Gemini Live API...")

    ws = websocket.WebSocket()
    ws.connect(REALTIME_URL)

    # Send setup message to initialize session
    setup_message = {
        "setup": {
            "model": "gemini-2.5-flash-native-audio-dialog",
            "generation_config": {
                "response_modalities": ["AUDIO", "TEXT"],
                "speech_config": {
                    "voice_config": {
                        "prebuilt_voice_config": {"voice_name": "Ava"}
                    }
                },
            },
        }
    }
    ws.send(json.dumps(setup_message))

    st.success("✅ Connected! Start singing...")

    # Threads: one sends audio, one receives feedback
    t1 = threading.Thread(target=stream_microphone_audio, args=(ws,))
    t2 = threading.Thread(target=receive_from_gemini, args=(ws,))
    t1.start()
    t2.start()

    # Stop button
    if st.button("🛑 Stop Session"):
        stop_flag.set()
        ws.close()
        st.success("✅ Session ended.")


# ========================================
# Streamlit UI
# ========================================
st.markdown("### 🎙️ Real-Time AI Feedback")
st.write("This mode streams your voice directly to Gemini and gives instant spoken + text feedback.")

if st.button("🚀 Start Live Coaching"):
    start_live_coaching()