import streamlit as st
import base64
import tempfile
import io
import asyncio
import soundfile as sf
from google import genai
import requests
import wave

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
uploaded = st.file_uploader("Upload audio (WAV/MP3/M4A)", type=["wav", "mp3", "m4a"])
if uploaded:
    file_bytes = uploaded.read()
    ext = uploaded.name.split('.')[-1].lower()
    if ext != "wav":
        audio_bytes = convert_to_wav_bytes(file_bytes)
    else:
        audio_bytes = file_bytes

    tmp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp_path = tmp_file.name
    with open(tmp_path, "wb") as f:
        f.write(audio_bytes)
    st.audio(tmp_path, format="audio/wav")

# -------------------------
# Helper: Corrected Gemini TTS using official API
# -------------------------
async def synthesize_speech(text_prompt, voice_name="Kore"):
    """
    Correct Gemini TTS API call using official documentation structure
    """
    url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-tts:generateContent"
    headers = {
        "x-goog-api-key": st.secrets['GOOGLE_API_KEY'],  # Correct header format
        "Content-Type": "application/json"
    }
    
    # Correct request structure per official documentation
    data = {
        "contents": [{
            "parts": [{
                "text": text_prompt
            }]
        }],
        "generationConfig": {
            "responseModalities": ["AUDIO"],
            "speechConfig": {
                "voiceConfig": {
                    "prebuiltVoiceConfig": {
                        "voiceName": voice_name
                    }
                }
            }
        }
    }
    
    loop = asyncio.get_event_loop()
    response = await loop.run_in_executor(
        None, lambda: requests.post(url, headers=headers, json=data)
    )
    response.raise_for_status()
    
    # Extract audio data from correct response structure
    response_json = response.json()
    audio_base64 = response_json["candidates"][0]["content"]["parts"][0]["inlineData"]["data"]
    
    if audio_base64 is None:
        raise ValueError("No audio returned from Gemini TTS.")
    
    return base64.b64decode(audio_base64)

# -------------------------
# Helper: Convert PCM to WAV
# -------------------------
def pcm_to_wav(pcm_data, channels=1, sample_rate=24000, sample_width=2):
    """Convert raw PCM data to WAV format"""
    wav_buffer = io.BytesIO()
    with wave.open(wav_buffer, 'wb') as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(sample_width) 
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm_data)
    return wav_buffer.getvalue()

# -------------------------
# Step 2 & 3: Transcribe & TTS with progress
# -------------------------
async def transcribe_and_sing():
    client = genai.Client()

    progress_text = st.empty()
    progress_bar = st.progress(0)

    # Estimate duration
    data, samplerate = sf.read(tmp_path, always_2d=True)
    duration = len(data) / samplerate
    step_transcribe = 50 / max(duration, 1)
    step_tts = 50 / max(duration, 1)

    # --- Transcription ---
    progress_text.text("Transcribing with Gemini 1.5 Flash...")
    try:
        resp = client.models.generate_content(
            model="gemini-1.5-flash",
            contents=[
                {"role": "user", "parts": [
                    {"text": "Please transcribe this speech."},
                    {"inline_data": {"mime_type": "audio/wav", "data": base64.b64encode(audio_bytes).decode()}}
                ]}
            ]
        )
        transcript = resp.text
    except Exception as e:
        st.error(f"Transcription failed: {e}")
        return

    # Simulate progress for transcription
    for i in range(int(duration)):
        progress_bar.progress(min(int((i + 1) * step_transcribe), 50))
        await asyncio.sleep(0.05)
    st.success("✅ Transcription complete!")
    st.write(transcript)

    # --- TTS with natural language prompt ---
    progress_text.text(f"Generating singing-style voice ({singing_style}) with Gemini 2.5 TTS...")
    
    # Use natural language prompt for style control
    tts_prompt = f"Sing these words in a {singing_style.lower()} style with emotion and expression: {transcript}"
    
    tts_task = asyncio.create_task(synthesize_speech(tts_prompt))
    for i in range(int(duration)):
        progress_bar.progress(min(50 + int((i + 1) * step_tts), 100))
        await asyncio.sleep(0.05)
    
    pcm_data = await tts_task
    vocal_bytes = pcm_to_wav(pcm_data)  # Convert PCM to WAV

    progress_bar.progress(100)
    progress_text.text("🎶 Your sung version is ready!")

    # Save vocal
    vocal_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    vocal_path = vocal_file.name
    with open(vocal_path, "wb") as f:
        f.write(vocal_bytes)

    st.audio(vocal_path, format="audio/wav")
    with open(vocal_path, "rb") as f:
        st.download_button("Download Vocal", f, file_name="singified.wav", mime="audio/wav")

# --- Trigger ---
if audio_bytes is not None and st.button("🎶 Transcribe & Sing"):
    asyncio.run(transcribe_and_sing())