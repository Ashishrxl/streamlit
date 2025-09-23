import streamlit as st
import base64
import tempfile
import io
import numpy as np
import asyncio
import soundfile as sf
from google import genai
import requests

st.set_page_config(page_title="Singify 🎶", layout="centered")
st.title("🎤 Singify with Gemini")
st.caption("Record or upload a line → Transcribe with Gemini 1.5 Flash → Sing it back with Gemini 2.5 TTS")

# Sidebar
singing_style = st.sidebar.selectbox("Singing Style", ["Pop", "Ballad", "Rap", "Soft"])

audio_bytes = None
tmp_path = None

# -------------------------
# Helper: Convert audio to WAV bytes
# -------------------------
def convert_to_wav_bytes(file_bytes):
    """
    Convert MP3/M4A/WAV audio bytes to WAV bytes using soundfile + audioread
    """
    with io.BytesIO(file_bytes) as f:
        data, samplerate = sf.read(f, always_2d=True)
    out_bytes = io.BytesIO()
    sf.write(out_bytes, data, samplerate, format='WAV')
    return out_bytes.getvalue()

# -------------------------
# Step 1: Upload audio
# -------------------------
uploaded = st.file_uploader("Upload audio (WAV/MP3/M4A)", type=["wav","mp3","m4a"])
if uploaded:
    file_bytes = uploaded.read()
    ext = uploaded.name.split('.')[-1].lower()
    if ext != "wav":
        audio_bytes = convert_to_wav_bytes(file_bytes)
    else:
        audio_bytes = file_bytes

    tmp_path = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
    with open(tmp_path, "wb") as f:
        f.write(audio_bytes)
    st.audio(tmp_path, format="audio/wav")

# -------------------------
# Helper: Gemini TTS
# -------------------------
async def synthesize_speech(ssml_text, voice="alloy"):
    url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-pro-preview-tts:generateSpeech"
    headers = {
        "Authorization": f"Bearer {st.secrets['gemini_api_key']}",
        "Content-Type": "application/json",
    }
    data = {"input": {"ssml": ssml_text}, "voice": voice, "audioFormat": "wav"}
    loop = asyncio.get_event_loop()
    response = await loop.run_in_executor(None, lambda: requests.post(url, headers=headers, json=data))
    response.raise_for_status()
    audio_base64 = response.json().get("audio")
    if audio_base64 is None:
        raise ValueError("No audio returned from Gemini TTS.")
    return base64.b64decode(audio_base64)

# -------------------------
# Step 2 & 3: Transcribe & TTS with progress
# -------------------------
async def transcribe_and_sing():
    client = genai.Client()
    
    progress_text = st.empty()
    progress_bar = st.progress(0)

    # Estimate duration
    data, samplerate = sf.read(tmp_path, always_2d=True)
    duration = len(data)/samplerate
    step_transcribe = 50/max(duration,1)
    step_tts = 50/max(duration,1)

    # --- Transcription ---
    progress_text.text("Transcribing with Gemini 1.5 Flash...")
    try:
        resp = client.models.generate_content(
            model="gemini-1.5-flash",
            contents=[{"role": "user", "parts":[
                {"text": "Please transcribe this speech."},
                {"inline_data": {"mime_type": "audio/wav", "data": base64.b64encode(audio_bytes).decode()}}
            ]}]
        )
        transcript = resp.text
    except Exception as e:
        st.error(f"Transcription failed: {e}")
        return

    # Simulate progress for transcription
    for i in range(int(duration)):
        progress_bar.progress(min(int((i+1)*step_transcribe),50))
        await asyncio.sleep(0.05)

    st.success("✅ Transcription complete!")
    st.write(transcript)

    # --- TTS ---
    progress_text.text(f"Generating singing-style voice ({singing_style}) with Gemini 2.5 TTS...")
    ssml = f"<speak><prosody rate='95%' pitch='+2st'>Sing these words in a {singing_style} style: {transcript}</prosody></speak>"

    tts_task = asyncio.create_task(synthesize_speech(ssml))
    for i in range(int(duration)):
        progress_bar.progress(min(50 + int((i+1)*step_tts), 100))
        await asyncio.sleep(0.05)
    vocal_bytes = await tts_task

    # Complete
    progress_bar.progress(100)
    progress_text.text("🎶 Your sung version is ready!")

    # Save vocal
    vocal_path = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
    with open(vocal_path, "wb") as f:
        f.write(vocal_bytes)

    st.audio(vocal_path, format="audio/wav")
    with open(vocal_path, "rb") as f:
        st.download_button("Download Vocal", f, file_name="singified.wav", mime="audio/wav")

# --- Trigger ---
if audio_bytes is not None and st.button("🎶 Transcribe & Sing"):
    asyncio.run(transcribe_and_sing())