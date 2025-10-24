import streamlit as st
import asyncio
import numpy as np
from streamlit_webrtc import webrtc_streamer, WebRtcMode
from google import genai
from google.genai import types

st.set_page_config(page_title="Voice Translator: Hindi ↔ English", page_icon="🌐", layout="wide")

st.title("🌐 Real-Time Hindi ↔ English Voice Translator")
st.markdown("**Powered by Google Gemini 2.5 Flash Native Audio**")

try:
    API_KEY = st.secrets["GOOGLE_API_KEY"]
except Exception:
    st.error("Missing 'google_api_key' in Streamlit secrets.")
    st.stop()

model = st.selectbox("Model", ["gemini-2.5-flash-native-audio-latest", "gemini-live-2.5-flash-preview"])
src_lang = st.selectbox("Source Language", ["Hindi (hi)", "English (en)"])
tgt_lang = st.selectbox("Target Language", ["English (en)", "Hindi (hi)"])

client = genai.Client(api_key=API_KEY)
config = types.LiveConnectConfig(
    response_modalities=["AUDIO"],
    system_instruction=f"You are a real-time translator between {src_lang} and {tgt_lang}."
)

st.session_state.session = client.aio.live.connect(model=model, config=config)

st.markdown("### Speak and translate live from your browser microphone.")

audio_buffer = []

def audio_frame_callback(frame):
    array = frame.to_ndarray(format="s16")
    audio_buffer.append(array.tobytes())
    return frame

webrtc_ctx = webrtc_streamer(
    key="translator",
    mode=WebRtcMode.SENDRECV,
    audio_frame_callback=audio_frame_callback,
    media_stream_constraints={"audio": True, "video": False},
    async_processing=True,
)

if st.button("Translate Spoken Audio"):
    if not audio_buffer:
        st.warning("Please speak first!")
    else:
        audio_bytes = b"".join(audio_buffer)
        audio_buffer.clear()

        async def translate_and_play():
            session = await st.session_state.session.__aenter__()
            await session.send_realtime_input(audio=types.Blob(data=audio_bytes, mime_type="audio/pcm;rate=16000"))
            translated_audio_chunks = []
            async for resp in session.receive():
                if resp.data:
                    translated_audio_chunks.append(resp.data)
                if resp.server_content and resp.server_content.turn_complete:
                    break
            await session.__aexit__(None, None, None)

            translated_audio = b"".join(translated_audio_chunks)
            st.audio(translated_audio, format="audio/wav")

        asyncio.run(translate_and_play())