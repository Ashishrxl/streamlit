import os
import asyncio
import json
import base64
import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode
import av
import numpy as np
import aiohttp

st.set_page_config(page_title="Voice Translator", layout="wide")
API_KEY = os.getenv("GOOGLE_API_KEY", "")
MODEL = "models/gemini-2.5-flash-native-audio-latest"
WS_ENDPOINT = "wss://generativelanguage.googleapis.com/ws/google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent"

st.title("🌐 Real-Time Voice Translator (Hindi ↔ English)")
st.caption("Runs on any browser or device — no PortAudio needed!")

if 'translations' not in st.session_state:
    st.session_state.translations = []

async def translate_audio(audio_bytes: bytes, target_lang: str):
    headers = {"Authorization": f"Bearer {API_KEY}"}
    async with aiohttp.ClientSession() as session:
        async with session.ws_connect(WS_ENDPOINT, headers=headers, max_msg_size=0) as ws:
            await ws.send_json({
                "model": MODEL,
                "generationConfig": {
                    "responseModalities": ["AUDIO"],
                    "speechConfig": {"voiceConfig": {"languageCode": target_lang}}
                },
                "systemInstruction": "Act as a real-time translator between Hindi and English."
            })
            await ws.send_json({"data": base64.b64encode(audio_bytes).decode("utf-8")})
            await ws.send_json({"turnComplete": True})
            result = b""
            async for msg in ws:
                data = json.loads(msg.data)
                if "data" in data:
                    result += base64.b64decode(data["data"])
                if data.get("serverContent", {}).get("turnComplete"):
                    break
            return result

def audio_callback(frame: av.AudioFrame):
    pcm = frame.to_ndarray().astype(np.int16).tobytes()
    st.session_state.audio_buffer = pcm
    return frame

rtc_config = {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
st.markdown("### 🗣️ Speak into your browser microphone")

col1, col2 = st.columns(2)
with col1:
    st.subheader("👤 User 1: Hindi → English")
    ctx_hi = webrtc_streamer(
        key="hindi",
        mode=WebRtcMode.SENDONLY,
        rtc_configuration=rtc_config,
        media_stream_constraints={"audio": True, "video": False},
        audio_frame_callback=audio_callback
    )
    if st.button("Translate from Hindi"):
        if 'audio_buffer' in st.session_state:
            translated_audio = asyncio.run(translate_audio(st.session_state.audio_buffer, "en-US"))
            st.audio(translated_audio, format="audio/wav")
            st.session_state.translations.append("Hindi → English done ✅")

with col2:
    st.subheader("👤 User 2: English → Hindi")
    ctx_en = webrtc_streamer(
        key="english",
        mode=WebRtcMode.SENDONLY,
        rtc_configuration=rtc_config,
        media_stream_constraints={"audio": True, "video": False},
        audio_frame_callback=audio_callback
    )
    if st.button("Translate from English"):
        if 'audio_buffer' in st.session_state:
            translated_audio = asyncio.run(translate_audio(st.session_state.audio_buffer, "hi-IN"))
            st.audio(translated_audio, format="audio/wav")
            st.session_state.translations.append("English → Hindi done ✅")

st.divider()
st.subheader("📜 Translation Log")
for t in st.session_state.translations[-10:]:
    st.write("• " + t)

st.sidebar.text_input("Google API Key", type="password", key="api_key")
st.sidebar.caption("This version runs in browser — mobile and desktop supported.")