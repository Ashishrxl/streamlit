import streamlit as st
import base64
import tempfile
import io
import asyncio
import soundfile as sf
from google import genai
import requests
import wave
import numpy as np

st.set_page_config(page_title="Singify 🎶", layout="centered")
st.title("🎤 Singify with Gemini")
st.caption("Record or upload a line → Transcribe with Gemini 1.5 Flash → Sing it back with Gemini 2.5 TTS")

# Sidebar
singing_style = st.sidebar.selectbox("Singing Style", ["Pop", "Ballad", "Rap", "Soft"])
voice_option = st.sidebar.selectbox("Voice", ["Kore", "Charon", "Fenrir", "Aoede"])

audio_bytes = None
tmp_path = None

# -------------------------
# Helper: Convert audio to WAV bytes
# -------------------------
def convert_to_wav_bytes(file_bytes):
    """
    Convert MP3/M4A/WAV audio bytes to WAV bytes using soundfile
    """
    try:
        with io.BytesIO(file_bytes) as f:
            data, samplerate = sf.read(f, always_2d=True)
        out_bytes = io.BytesIO()
        sf.write(out_bytes, data, samplerate, format='WAV')
        return out_bytes.getvalue()
    except Exception as e:
        st.error(f"Error converting audio: {e}")
        return None

# -------------------------
# Audio Input Options
# -------------------------
st.subheader("📤 Choose Audio Input Method")

# Create tabs for different input methods
tab1, tab2 = st.tabs(["📁 Upload Audio File", "🎙️ Record Audio"])

with tab1:
    st.markdown("**Upload an audio file from your device**")
    uploaded = st.file_uploader(
        "Choose an audio file", 
        type=["wav", "mp3", "m4a", "ogg", "flac"],
        help="Supported formats: WAV, MP3, M4A, OGG, FLAC (Max 200MB)"
    )
    
    if uploaded:
        st.success(f"✅ Uploaded: {uploaded.name} ({uploaded.size / 1024 / 1024:.2f} MB)")
        file_bytes = uploaded.read()
        ext = uploaded.name.split('.')[-1].lower()
        
        if ext != "wav":
            with st.spinner("Converting audio to WAV..."):
                audio_bytes = convert_to_wav_bytes(file_bytes)
        else:
            audio_bytes = file_bytes
        
        if audio_bytes:
            tmp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            tmp_path = tmp_file.name
            with open(tmp_path, "wb") as f:
                f.write(audio_bytes)
            
            # Show audio info
            data, samplerate = sf.read(tmp_path, always_2d=True)
            duration = len(data) / samplerate
            st.info(f"🎵 Duration: {duration:.2f}s | Sample Rate: {samplerate} Hz | Channels: {data.shape[1]}")
            st.audio(tmp_path, format="audio/wav")

with tab2:
    st.markdown("**Record audio directly in your browser**")
    
    # Simple recording interface
    if st.button("🎙️ Start Recording (Click and speak, then click Stop)"):
        st.info("🔴 Recording... Click 'Stop Recording' when done")
    
    # Note: For actual recording, you would need streamlit-audio-recorder
    # pip install streamlit-audio-recorder
    try:
        from audio_recorder_streamlit import audio_recorder
        
        recorded_audio = audio_recorder(
            text="Click to record",
            recording_color="#e8b62c",
            neutral_color="#6aa36f",
            icon_name="microphone-lines",
            icon_size="2x",
        )
        
        if recorded_audio is not None:
            st.success("✅ Audio recorded successfully!")
            audio_bytes = recorded_audio
            
            tmp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            tmp_path = tmp_file.name
            with open(tmp_path, "wb") as f:
                f.write(audio_bytes)
            
            # Show recorded audio info
            data, samplerate = sf.read(tmp_path, always_2d=True)
            duration = len(data) / samplerate
            st.info(f"🎵 Duration: {duration:.2f}s | Sample Rate: {samplerate} Hz")
            st.audio(tmp_path, format="audio/wav")
            
    except ImportError:
        st.warning("📝 To enable recording,")
        st.code("pip", language="bash")

# -------------------------
# Additional Upload Options
# -------------------------
st.subheader("📎 Additional Options")

col1, col2 = st.columns(2)
with col1:
    if st.button("🗑️ Clear Audio"):
        audio_bytes = None
        tmp_path = None
        st.rerun()

with col2:
    if audio_bytes and st.button("ℹ️ Audio Info"):
        data, samplerate = sf.read(tmp_path, always_2d=True)
        duration = len(data) / samplerate
        file_size = len(audio_bytes) / 1024 / 1024
        
        st.info(f"""
        **Audio Information:**
        - Duration: {duration:.2f} seconds
        - Sample Rate: {samplerate} Hz
        - Channels: {data.shape[1]}
        - File Size: {file_size:.2f} MB
        - Format: WAV
        """)

# -------------------------
# Helper: Corrected Gemini TTS using official API
# -------------------------
async def synthesize_speech(text_prompt, voice_name="Kore"):
    """
    Correct Gemini TTS API call using official documentation structure
    """
    url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-tts:generateContent"
    headers = {
        "x-goog-api-key": st.secrets['GOOGLE_API_KEY'],
        "Content-Type": "application/json"
    }
    
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
    progress_text.text("🔤 Transcribing with Gemini 1.5 Flash...")
    try:
        resp = client.models.generate_content(
            model="gemini-1.5-flash",
            contents=[
                {"role": "user", "parts": [
                    {"text": "Please transcribe this speech accurately."},
                    {"inline_data": {"mime_type": "audio/wav", "data": base64.b64encode(audio_bytes).decode()}}
                ]}
            ]
        )
        transcript = resp.text.strip()
    except Exception as e:
        st.error(f"❌ Transcription failed: {e}")
        return

    # Simulate progress for transcription
    for i in range(int(max(duration, 1))):
        progress_bar.progress(min(int((i + 1) * step_transcribe), 50))
        await asyncio.sleep(0.05)
        
    st.success("✅ Transcription complete!")
    st.write(f"**Transcribed Text:** {transcript}")

    # --- TTS with natural language prompt ---
    progress_text.text(f"🎵 Generating {singing_style} style voice with Gemini 2.5 TTS...")
    
    tts_prompt = f"Sing these words in a {singing_style.lower()} style with emotion and musical expression: {transcript}"
    
    try:
        tts_task = asyncio.create_task(synthesize_speech(tts_prompt, voice_name=voice_option))
        for i in range(int(max(duration, 1))):
            progress_bar.progress(min(50 + int((i + 1) * step_tts), 100))
            await asyncio.sleep(0.05)
        
        pcm_data = await tts_task
        vocal_bytes = pcm_to_wav(pcm_data)

        progress_bar.progress(100)
        progress_text.text("🎶 Your sung version is ready!")

        # Save vocal
        vocal_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        vocal_path = vocal_file.name
        with open(vocal_path, "wb") as f:
            f.write(vocal_bytes)

        st.success("🎤 Generated singing voice!")
        st.audio(vocal_path, format="audio/wav")
        
        # Download buttons
        col1, col2 = st.columns(2)
        with col1:
            with open(vocal_path, "rb") as f:
                st.download_button(
                    "📥 Download Sung Version", 
                    f, 
                    file_name=f"singified_{singing_style.lower()}.wav", 
                    mime="audio/wav"
                )
        with col2:
            with open(tmp_path, "rb") as f:
                st.download_button(
                    "📥 Download Original", 
                    f, 
                    file_name="original_audio.wav", 
                    mime="audio/wav"
                )
                
    except Exception as e:
        st.error(f"❌ TTS generation failed: {e}")
        progress_text.text("❌ Generation failed")

# -------------------------
# Main Process Button
# -------------------------
st.subheader("🚀 Generate Singing Voice")

if audio_bytes is not None:
    if st.button("🎶 Transcribe & Sing", type="primary"):
        asyncio.run(transcribe_and_sing())
else:
    st.warning("⚠️ Please upload or record an audio file first!")
    
# -------------------------
# Instructions
# -------------------------
st.subheader("📋 How to Use")
st.markdown("""
1. **Upload** an audio file or **record** your voice
2. Choose your preferred **singing style** and **voice** from the sidebar
3. Click **"Transcribe & Sing"** to process
4. Download your **sung version** and **original audio**

**Supported Formats:** WAV, MP3, M4A, OGG, FLAC  
**Max File Size:** 200MB  
**Best Results:** Clear speech, minimal background noise
""")
