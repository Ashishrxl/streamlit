import streamlit as st
import asyncio
import numpy as np
import sounddevice as sd
from google import genai
from google.genai import types

st.set_page_config(page_title="Voice Translator: Hindi ↔ English", page_icon="🌐", layout="wide")

if "translator" not in st.session_state:
    st.session_state.translator = None
if "session" not in st.session_state:
    st.session_state.session = None
if "connected" not in st.session_state:
    st.session_state.connected = False

st.title("🌐 Real-Time Hindi ↔ English Voice Translator")
st.markdown("**Powered by Google Gemini 2.5 Flash Native Audio**")

# Read API key from Streamlit secrets
try:
    key = st.secrets["GOOGLE_API_KEY"]
except Exception:
    st.error("Google API Key not found in Streamlit secrets. Please add `google_api_key` to your secrets.")
    st.stop()

model = st.selectbox("Model", ["gemini-2.5-flash-native-audio-latest", "gemini-live-2.5-flash-preview"])
src = st.selectbox("Source Language", ["Hindi (hi)", "English (en)"])
tgt = st.selectbox("Target Language", ["English (en)", "Hindi (hi)"])

if not st.session_state.connected:
    client = genai.Client(api_key=key)
    config = types.LiveConnectConfig(
        response_modalities=["AUDIO"],
        system_instruction=f"You are a real-time bidirectional translator between {src} and {tgt}."
    )
    session = client.aio.live.connect(model=model, config=config)
    st.session_state.client = client
    st.session_state.session = session
    st.session_state.connected = True
    st.success("Connected to Gemini API")

async def record_and_translate(lang_code):
    samplerate = 16000
    duration = 5
    st.info("Recording...")
    audio = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype='int16')
    sd.wait()
    audio_bytes = audio.tobytes()

    async with st.session_state.session as session:
        await session.send_realtime_input(audio=types.Blob(data=audio_bytes, mime_type="audio/pcm;rate=16000"))
        st.info("Translating...")
        translated_audio = []
        async for response in session.receive():
            if response.data:
                translated_audio.append(response.data)
            if response.server_content and response.server_content.turn_complete:
                break
        audio_out = b''.join(translated_audio)
        st.success("Playing Translated Audio")
        np_audio = np.frombuffer(audio_out, dtype=np.int16)
        sd.play(np_audio, 24000)
        sd.wait()

col1, col2 = st.columns(2)

with col1:
    if st.button("🎤 Speak in Source Language"):
        asyncio.run(record_and_translate(src))

with col2:
    if st.button("🎤 Speak in Target Language"):
        asyncio.run(record_and_translate(tgt))