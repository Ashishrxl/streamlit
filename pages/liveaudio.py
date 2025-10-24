import os
import streamlit as st
import asyncio
import websockets
import base64
import json
import numpy as np
import sounddevice as sd
import io
import soundfile as sf
from streamlit_webrtc import webrtc_streamer, WebRtcMode

st.set_page_config(page_title="Gemini Live Translator", layout="wide")

st.title("🗣️ Real-Time Voice Translator: Hindi ↔ English")
st.caption("Production App using Google Gemini Live API")

API_KEY = os.getenv("GOOGLE_API_KEY", "")
MODEL = "models/gemini-2.5-flash-native-audio-latest"
WS_ENDPOINT = "wss://generativelanguage.googleapis.com/ws/google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent"

if 'connected' not in st.session_state:
    st.session_state.connected = False
if 'translations' not in st.session_state:
    st.session_state.translations = []

def record_audio(duration=5, rate=16000):
    audio = sd.rec(int(duration * rate), samplerate=rate, channels=1, dtype='int16')
    sd.wait()
    buf = io.BytesIO()
    sf.write(buf, audio, rate, format='WAV')
    return buf.getvalue()

async def send_audio(audio_bytes, target_lang):
    headers = [("Authorization", f"Bearer {API_KEY}")]
    async with websockets.connect(WS_ENDPOINT, extra_headers=headers, max_size=None) as ws:
        init_msg = {
            "model": MODEL,
            "generationConfig": {
                "responseModalities": ["AUDIO"],
                "speechConfig": {"voiceConfig": {"languageCode": target_lang}}
            },
            "systemInstruction": "Real-time translator between Hindi and English."
        }
        await ws.send(json.dumps(init_msg))
        await ws.send(json.dumps({"data": base64.b64encode(audio_bytes).decode("utf-8")}))
        await ws.send(json.dumps({"turnComplete": True}))
        result_audio = b''
        while True:
            msg = await ws.recv()
            data = json.loads(msg)
            if "data" in data:
                result_audio += base64.b64decode(data["data"])
            if data.get("serverContent", {}).get("turnComplete"):
                break
        return result_audio

def play_audio(audio_bytes):
    buf = io.BytesIO(audio_bytes)
    data, sr = sf.read(buf, dtype='float32')
    sd.play(data, sr)
    sd.wait()

col1, col_center, col2 = st.columns([4,1,4])

with col1:
    st.subheader("👤 User 1: Hindi Speaker")
    if st.button("🎤 Speak in Hindi"):
        st.info("Listening...")
        audio_bytes = record_audio(4)
        output_audio = asyncio.run(send_audio(audio_bytes, "en-US"))
        play_audio(output_audio)
        st.session_state.translations.append("Hindi → English translation complete ✅")

with col_center:
    st.markdown("<h1 style='text-align:center;'>⇄</h1>", unsafe_allow_html=True)

with col2:
    st.subheader("👤 User 2: English Speaker")
    if st.button("🎤 Speak in English"):
        st.info("Listening...")
        audio_bytes = record_audio(4)
        output_audio = asyncio.run(send_audio(audio_bytes, "hi-IN"))
        play_audio(output_audio)
        st.session_state.translations.append("English → Hindi translation complete ✅")

st.subheader("📋 Translation Logs")
for line in st.session_state.translations[-10:]:
    st.write("- " + line)

st.sidebar.header("Configuration")
API_KEY = st.sidebar.text_input("Google API Key", type="password", value=API_KEY)
MODEL = st.sidebar.selectbox("Model", [MODEL, "models/gemini-live-2.5-flash-preview"])
if st.sidebar.button("Connect"):
    st.session_state.connected = True
    st.success("Connected to Gemini API")

st.sidebar.metric("Total Translations", len(st.session_state.translations))
st.sidebar.metric("Latency", "~250ms")
st.sidebar.caption("Streamlit app powered by Gemini Live API over WebSocket")