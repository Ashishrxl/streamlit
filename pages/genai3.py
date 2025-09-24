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
st.title("🎤 Singify")
st.caption("Record or upload a line → Transcribe....")

# Initialize session state
if 'transcript' not in st.session_state:
    st.session_state.transcript = None
if 'vocal_path' not in st.session_state:
    st.session_state.vocal_path = None
if 'original_path' not in st.session_state:
    st.session_state.original_path = None
if 'generation_complete' not in st.session_state:
    st.session_state.generation_complete = False
if 'current_style' not in st.session_state:
    st.session_state.current_style = None
if 'current_voice' not in st.session_state:
    st.session_state.current_voice = None
if 'recording_state' not in st.session_state:
    st.session_state.recording_state = "idle"  # idle, recording, recorded

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

            # Store original path in session state
            st.session_state.original_path = tmp_path

            # Show audio info
            data, samplerate = sf.read(tmp_path, always_2d=True)
            duration = len(data) / samplerate
            st.info(f"🎵 Duration: {duration:.2f}s | Sample Rate: {samplerate} Hz | Channels: {data.shape[1]}")
            st.audio(tmp_path, format="audio/wav")

with tab2:
    st.markdown("**Record audio directly in your browser**")

    # Check if audio_recorder_streamlit is available
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

            # Store original path in session state
            st.session_state.original_path = tmp_path

            # Show recorded audio info
            data, samplerate = sf.read(tmp_path, always_2d=True)
            duration = len(data) / samplerate
            st.info(f"🎵 Duration: {duration:.2f}s | Sample Rate: {samplerate} Hz")
            st.audio(tmp_path, format="audio/wav")

    except ImportError:
        st.warning("⚠️ Audio recording requires the `streamlit-audio-recorder` package.")
        st.info("Install it with: `pip install streamlit-audio-recorder`")
        
        # Alternative manual recording interface with proper start/stop buttons
        
        st.markdown("**Alternative Recording Method:**")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.session_state.recording_state == "idle":
                if st.button("🎙️ Start Recording", type="primary"):
                    st.session_state.recording_state = "recording"
                    try:
                        st.experimental_rerun()
                    except AttributeError:
                        st.stop()
            
            elif st.session_state.recording_state == "recording":
                if st.button("⏹️ Stop Recording", type="secondary"):
                    st.session_state.recording_state = "recorded"
                    try:
                        st.experimental_rerun()
                    except AttributeError:
                        st.stop()
        
        with col2:
            if st.session_state.recording_state in ["recording", "recorded"]:
                if st.button("🗑️ Clear Recording"):
                    st.session_state.recording_state = "idle"
                    try:
                        st.experimental_rerun()
                    except AttributeError:
                        st.stop()
        
        # Show recording status
        if st.session_state.recording_state == "recording":
            st.error("🔴 **Recording in progress...** Click 'Stop Recording' when finished")
            st.info("💡 **Note:** This is a placeholder interface. For actual recording functionality, please install `streamlit-audio-recorder`")
        
        elif st.session_state.recording_state == "recorded":
            st.success("✅ **Recording completed!**")
            st.info("💡 **Note:** This is a placeholder recording. To enable actual audio recording, install `streamlit-audio-recorder` package")

# -------------------------
# Additional Upload Options
# -------------------------
st.subheader("📎 Additional Options")

col1, col2, col3 = st.columns(3)
with col1:
    if st.button("🗑️ Clear Audio"):
        audio_bytes = None
        tmp_path = None
        # Clear session state
        st.session_state.transcript = None
        st.session_state.vocal_path = None
        st.session_state.original_path = None
        st.session_state.generation_complete = False
        st.session_state.current_style = None
        st.session_state.current_voice = None
        st.session_state.recording_state = "idle"
        try:
            st.experimental_rerun()
        except AttributeError:
            st.stop()

with col2:
    if (audio_bytes or st.session_state.original_path or st.session_state.recording_state == "recorded") and st.button("ℹ️ Audio Info"):
        path_to_check = tmp_path if tmp_path else st.session_state.original_path
        if path_to_check:
            data, samplerate = sf.read(path_to_check, always_2d=True)
            duration = len(data) / samplerate
            file_size = len(audio_bytes) / 1024 / 1024 if audio_bytes else 0

            st.info(f"""
            **Audio Information:**
            - Duration: {duration:.2f} seconds
            - Sample Rate: {samplerate} Hz
            - Channels: {data.shape[1]}
            - File Size: {file_size:.2f} MB
            - Format: WAV
            """)
        else:
            st.info("**Placeholder Recording Information:**
- This is a demo recording interface
- Install `streamlit-audio-recorder` for actual recording")

with col3:
    if st.session_state.generation_complete and st.button("🆕 Generate New"):
        st.session_state.transcript = None
        st.session_state.vocal_path = None
        st.session_state.generation_complete = False
        st.session_state.current_style = None
        st.session_state.current_voice = None
        try:
            st.experimental_rerun()
        except AttributeError:
            st.stop()

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
        raise ValueError("No audio returned.")

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

    # Use tmp_path or stored original_path
    audio_path = tmp_path if tmp_path else st.session_state.original_path
    
    # Handle placeholder recording case
    if not audio_path and st.session_state.recording_state == "recorded":
        st.error("❌ This is a placeholder recording. Please install `streamlit-audio-recorder` for actual recording functionality.")
        return
    
    if not audio_path:
        st.error("No audio file available")
        return

    # Read audio for processing
    if audio_bytes:
        current_audio_bytes = audio_bytes
    else:
        with open(audio_path, "rb") as f:
            current_audio_bytes = f.read()

    # Estimate duration
    data, samplerate = sf.read(audio_path, always_2d=True)
    duration = len(data) / samplerate
    step_transcribe = 50 / max(duration, 1)
    step_tts = 50 / max(duration, 1)

    # --- Transcription ---
    progress_text.text("🔤 Transcribing...")
    try:
        resp = client.models.generate_content(
            model="gemini-1.5-flash",
            contents=[
                {"role": "user", "parts": [
                    {"text": "Please transcribe this speech accurately."},
                    {"inline_data": {"mime_type": "audio/wav", "data": base64.b64encode(current_audio_bytes).decode()}}
                ]}
            ]
        )
        transcript = resp.text.strip()
        st.session_state.transcript = transcript  # Store in session state
    except Exception as e:
        st.error(f"❌ Transcription failed: {e}")
        return

    # Simulate progress for transcription
    for i in range(int(max(duration, 1))):
        progress_bar.progress(min(int((i + 1) * step_transcribe), 50))
        await asyncio.sleep(0.05)

    st.success("✅ Transcription complete!")

    # --- TTS with natural language prompt ---
    progress_text.text(f"🎵 Generating... {singing_style}")

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

        # Save vocal and store in session state
        vocal_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        vocal_path = vocal_file.name
        with open(vocal_path, "wb") as f:
            f.write(vocal_bytes)

        # Store results in session state
        st.session_state.vocal_path = vocal_path
        st.session_state.generation_complete = True
        st.session_state.current_style = singing_style
        st.session_state.current_voice = voice_option

    except Exception as e:
        st.error(f"❌ Generation failed: {e}")
        progress_text.text("❌ Generation failed")

# -------------------------
# Display Results (Persistent)
# -------------------------
def display_results():
    """Display results from session state"""
    if st.session_state.transcript:
        st.subheader("📝 Transcription Results")
        st.write(f"**Transcribed Text:** {st.session_state.transcript}")

    if st.session_state.generation_complete and st.session_state.vocal_path:
        st.subheader("🎶 Generated Singing Voice")
        st.success(f"🎤 Generated {st.session_state.current_style} style with {st.session_state.current_voice} voice!")

        # Display audio player
        st.audio(st.session_state.vocal_path, format="audio/wav")

        # Download buttons
        col1, col2 = st.columns(2)
        with col1:
            with open(st.session_state.vocal_path, "rb") as f:
                st.download_button(
                    "📥 Download New Version", 
                    f.read(), 
                    file_name=f"singified_{st.session_state.current_style.lower()}.wav", 
                    mime="audio/wav",
                    key="download_sung"
                )
        with col2:
            if st.session_state.original_path:
                with open(st.session_state.original_path, "rb") as f:
                    st.download_button(
                        "📥 Download Old Version", 
                        f.read(), 
                        file_name="original_audio.wav", 
                        mime="audio/wav",
                        key="download_original"
                    )

# -------------------------
# Main Process Button
# -------------------------
st.subheader("🚀 Generate Singing Voice")

# Modified condition to include placeholder recording state
if audio_bytes is not None or st.session_state.original_path or st.session_state.recording_state == "recorded":
    if not st.session_state.generation_complete:
        if st.button("🎶 Transcribe & Sing", type="primary"):
            # Handle placeholder recording case
            if st.session_state.recording_state == "recorded" and not st.session_state.original_path:
                st.warning("⚠️ This is a placeholder recording interface. Please install `streamlit-audio-recorder` for actual recording functionality, or upload an audio file instead.")
            else:
                asyncio.run(transcribe_and_sing())
    else:
        st.info("✅ Generation already completed! Results shown below.")
else:
    st.warning("⚠️ Please upload or record an audio file first!")

# -------------------------
# Always Display Results if Available
# -------------------------
display_results()

# -------------------------
# Instructions
# -------------------------
st.subheader("📋 How to Use")
st.markdown("""
1. **Upload** an audio file or **record** your voice
2. Choose your preferred **singing style** and **voice** from the sidebar
3. Click **"Transcribe & Sing"** to process
4. Download your **sung version** and **original audio**
5. Use **"Generate New"** to create different versions

**Supported Formats:** WAV, MP3, M4A, OGG, FLAC  
**Max File Size:** 200MB  
**Best Results:** Clear speech, minimal background noise

**For Recording:** Install `streamlit-audio-recorder` package for full recording functionality:""")