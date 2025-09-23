import streamlit as st
import base64
import tempfile
import requests
import io
from google import genai
from pydub import AudioSegment
import asyncio

# -------------------------
# App Config
# -------------------------
st.set_page_config(page_title="Singify 🎶", layout="centered")
st.title("🎤 Singify with Gemini")
st.caption("Record or upload a line → Transcribe with Gemini 1.5 Flash → Sing it back with Gemini 2.5 TTS")

# Sidebar: Singing style
singing_style = st.sidebar.selectbox("Singing Style", ["Pop", "Ballad", "Rap", "Soft"])

audio_bytes = None
tmp_path = None

# -------------------------
# Helper: Convert to WAV
# -------------------------
def convert_to_wav(input_bytes, input_format):
    """Convert audio bytes to WAV format using pydub."""
    audio = AudioSegment.from_file(io.BytesIO(input_bytes), format=input_format)
    tmp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
    audio.export(tmp_wav, format="wav")
    return tmp_wav

# -------------------------
# Step 1: Record/Upload Audio
# -------------------------
audio = st.audio_input("Record your audio")
if audio:
    audio_bytes = audio.getvalue()
    tmp_path = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
    with open(tmp_path, "wb") as f:
        f.write(audio_bytes)
    st.audio(tmp_path, format="audio/wav")

uploaded = st.file_uploader("Or upload your audio file", type=["wav", "mp3", "m4a"])
if uploaded:
    file_bytes = uploaded.read()
    file_ext = uploaded.name.split(".")[-1].lower()
    if file_ext != "wav":
        tmp_path = convert_to_wav(file_bytes, file_ext)
    else:
        tmp_path = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
        with open(tmp_path, "wb") as f:
            f.write(file_bytes)
    audio_bytes = open(tmp_path, "rb").read()
    st.audio(tmp_path, format="audio/wav")

# -------------------------
# Helper: Gemini 2.5 TTS
# -------------------------
async def synthesize_speech(ssml_text, voice="alloy"):
    url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-pro-preview-tts:generateSpeech"
    headers = {
        "Authorization": f"Bearer {st.secrets['GOOGLE_API_KEY']}",
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
# Step 2 & 3: Async Transcribe & TTS with Progress
# -------------------------
async def transcribe_and_sing():
    client = genai.Client()
    
    progress_text = st.empty()
    progress_bar = st.progress(0)

    # Estimate audio duration
    audio_segment = AudioSegment.from_file(tmp_path, format="wav")
    duration = audio_segment.duration_seconds
    step_transcribe = 50 / max(duration, 1)
    step_tts = 50 / max(duration, 1)

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

    # Complete progress
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