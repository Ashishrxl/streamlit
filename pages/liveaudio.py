import streamlit as st
import asyncio
import numpy as np
from google import genai
from google.genai import types
from streamlit_webrtc import webrtc_streamer, WebRtcMode
import av

st.set_page_config(page_title="Voice Translator", layout="wide")

st.title("🌐 Real-Time Hindi ↔ English Voice Translator (Gemini 2.5)")
st.caption("Works on desktop and mobile browsers — no PortAudio errors!")

try:
    API_KEY = st.secrets["GOOGLE_API_KEY"]
except Exception:
    st.error("Missing 'google_api_key' in Streamlit secrets. Add it to .streamlit/secrets.toml.")
    st.stop()

model = st.selectbox("Select Model", ["gemini-2.5-flash-native-audio-latest", "gemini-live-2.5-flash-preview"])
src_lang = st.selectbox("Source Language", ["Hindi (hi)", "English (en)"])
tgt_lang = st.selectbox("Target Language", ["English (en)", "Hindi (hi)"])

client = genai.Client(api_key=API_KEY)
config = types.LiveConnectConfig(
    response_modalities=["AUDIO"],
    system_instruction=f"You are a real-time translator between {src_lang} and {tgt_lang}."
)

st.session_state.session = client.aio.live.connect(model=model, config=config)

if "webrtc_ctx" not in st.session_state:
    st.session_state.webrtc_ctx = None
if "audio_buffer" not in st.session_state:
    st.session_state.audio_buffer = []

def audio_callback(frame: av.AudioFrame):
    pcm = frame.to_ndarray().astype(np.int16).tobytes()
    st.session_state.audio_buffer.append(pcm)
    return frame

st.markdown("### 🎙 Speak into your microphone below to translate audio live")

webrtc_ctx = webrtc_streamer(
    key="voice_translator",
    mode=WebRtcMode.SENDRECV,
    audio_frame_callback=audio_callback,
    media_stream_constraints={"audio": True, "video": False},
    async_processing=True
)
st.session_state.webrtc_ctx = webrtc_ctx

async def translate_audio(audio_data: bytes):
    async with st.session_state.session as session:
        await session.send_realtime_input(audio=types.Blob(data=audio_data, mime_type="audio/pcm;rate=16000"))
        translated_audio = []
        async for r in session.receive():
            if r.data:
                translated_audio.append(r.data)
            if getattr(r, "server_content", None) and r.server_content.turn_complete:
                break
        return b"".join(translated_audio)

col1, col2 = st.columns(2)
with col1:
    if st.button("🎤 Translate from Source"):
        if st.session_state.audio_buffer:
            data = b"".join(st.session_state.audio_buffer)
            st.session_state.audio_buffer.clear()
            translated = asyncio.run(translate_audio(data))
            st.audio(translated, format="audio/wav")
        else:
            st.warning("Speak into the mic first.")

with col2:
    if st.button("🛑 Stop and Reset Stream"):
        if st.session_state.webrtc_ctx and st.session_state.webrtc_ctx.state.playing:
            st.session_state.webrtc_ctx.stop()
        st.session_state.audio_buffer.clear()
        st.success("Microphone and streaming stopped safely.")