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

# UI tweak to hide external links
from streamlit.components.v1 import html
html(
  """
  <script>
  try {
    const sel = window.top.document.querySelectorAll('[href*="streamlit.io"], [href*="streamlit.app"]');
    sel.forEach(e => e.style.display='none');
  } catch(e) { console.warn('parent DOM not reachable', e); }
  </script>
  """,
  height=0
)

hide_streamlit_style = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
[data-testid="stStatusWidget"] {display: none;}
[data-testid="stToolbar"] {display: none;}
a[href^="https://github.com"] {display: none !important;}
a[href^="https://streamlit.io"] {display: none !important;}
header > div:nth-child(2) {
    display: none;
}
</style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

st.set_page_config(page_title="Singify 🎶", layout="centered")
st.title("🎤 Singify")
st.caption("Record or upload a line → Transcribe....")

# --- API Key selection ---
api_keys = {
    "Key 1": st.secrets["KEY_1"],
    "Key 2": st.secrets["KEY_2"], "Key 3": st.secrets["KEY_3"], "Key 4": st.secrets["KEY_4"], "Key 5": st.secrets["KEY_5"], "Key 6": st.secrets["KEY_6"], "Key 7": st.secrets["KEY_7"], "Key 8": st.secrets["KEY_8"], "Key 9": st.secrets["KEY_9"], "Key 10": st.secrets["KEY_10"], "Key 11": st.secrets["KEY_11"]
}
selected_key_name = st.selectbox("Select Key", list(api_keys.keys()))
api_key = api_keys[selected_key_name]

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
# Helper: Synthesize speech (sync) using official genai client
# -------------------------
def synthesize_speech_sync(text_prompt, voice_name="Kore", model_name="gemini-2.5-flash-preview-tts"):
    """
    Use google.genai client to generate TTS audio for `text_prompt`.
    Returns raw PCM bytes (usually int16 little-endian PCM, sample rate 24000).
    This function is synchronous so it can be called from both sync and async contexts (use run_in_executor in async flows).
    """
    # create client with api_key provided by the user (fixes missing-key error)
    client = genai.Client(api_key=api_key)

    # Import types from google.genai to construct config in the officially supported shape
    try:
        from google.genai import types
    except Exception:
        # Fallback to REST if types aren't available
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent"
        headers = {"x-goog-api-key": api_key, "Content-Type": "application/json"}
        data = {
            "contents": [{"parts": [{"text": text_prompt}]}],
            "generationConfig": {
                "responseModalities": ["AUDIO"],
                "speechConfig": {
                    "voiceConfig": {
                        "prebuiltVoiceConfig": {
                            "voiceName": voice_name
                        }
                    }
                }
            },
            "model": model_name
        }
        resp = requests.post(url, headers=headers, json=data)
        resp.raise_for_status()
        resp_json = resp.json()
        # path per docs: candidates[0].content.parts[0].inlineData.data
        try:
            data_field = resp_json["candidates"][0]["content"]["parts"][0]["inlineData"]["data"]
        except KeyError:
            data_field = resp_json["candidates"][0]["content"][0]["parts"][0]["inlineData"]["data"]

        if isinstance(data_field, str):
            return base64.b64decode(data_field)
        else:
            return bytes(data_field)

    # If we have types available, call the client properly (recommended)
    response = client.models.generate_content(
        model=model_name,
        contents=[{"parts": [{"text": text_prompt}]}],
        config=types.GenerateContentConfig(
            response_modalities=["AUDIO"],
            speech_config=types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(
                        voice_name=voice_name
                    )
                )
            ),
        )
    )

    # Defensive parsing of response
    try:
        data_field = response.candidates[0].content.parts[0].inline_data.data
    except Exception:
        try:
            data_field = response.candidates[0].content.parts[0].inlineData.data
        except Exception as e:
            raise RuntimeError(f"Unexpected response structure from TTS model: {e}")

    if isinstance(data_field, str):
        return base64.b64decode(data_field)
    else:
        return data_field

# -------------------------
# Helper: Convert PCM to WAV
# -------------------------
def pcm_to_wav(pcm_data, channels=1, sample_rate=24000, sample_width=2):
    """Convert raw PCM data (bytes) to WAV format bytes"""
    wav_buffer = io.BytesIO()
    with wave.open(wav_buffer, 'wb') as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(sample_width)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm_data)
    return wav_buffer.getvalue()

# -------------------------
# Initialize session state
# -------------------------
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

# Sidebar
singing_style = st.selectbox("Singing Style", ["Pop", "Ballad", "Rap", "Soft"])
voice_option = st.selectbox("Voice", ["Kore", "Charon", "Fenrir", "Aoede"])

audio_bytes = None
tmp_path = None

# -------------------------
# Audio Input Options
# -------------------------
st.subheader("📤 Choose Audio Input Method")

# Create tabs for different input methods (including the new text/doc tab)
tab1, tab2, tab3 = st.tabs(["📁 Upload Audio File", "🎙️ Record Audio", "📝 Upload Text or Document"])

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
    st.markdown("")

    # Option 1: Native Streamlit Audio Input (Recommended)
    st.markdown("")
    recorded_audio_native = st.audio_input("🎙️ Record your voice", key="native_recorder")

    if recorded_audio_native is not None:
        st.success("✅ Audio recorded successfully with native recorder!")

        # Read the audio bytes
        audio_bytes = recorded_audio_native.read()
        recorded_audio_native.seek(0)

        # Save to tmp file
        tmp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp_path = tmp_file.name
        with open(tmp_path, "wb") as f:
            f.write(audio_bytes)

        st.session_state.original_path = tmp_path

        try:
            data, samplerate = sf.read(tmp_path, always_2d=True)
            duration = len(data) / samplerate
            st.info(f"🎵 Duration: {duration:.2f}s | Sample Rate: {samplerate} Hz")
            st.audio(tmp_path, format="audio/wav")
        except Exception as e:
            st.warning(f"Could not read audio properties: {e}")
            st.audio(recorded_audio_native, format="audio/wav")

    st.markdown("---")

    # Option 2: Enhanced recorder from streamlit-audio-recorder package
    st.markdown("")

    try:
        from streamlit_audio_recorder import audio_recorder

        recorded_audio_enhanced = audio_recorder(
            text="Click to record",
            recording_color="#e8b62c",
            neutral_color="#6aa36f",
            icon_name="microphone",
            icon_size="2x",
            key="enhanced_recorder"
        )

        if recorded_audio_enhanced is not None:
            st.success("✅ Audio recorded successfully with enhanced recorder!")
            audio_bytes = recorded_audio_enhanced

            tmp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            tmp_path = tmp_file.name
            with open(tmp_path, "wb") as f:
                f.write(audio_bytes)

            st.session_state.original_path = tmp_path

            try:
                data, samplerate = sf.read(tmp_path, always_2d=True)
                duration = len(data) / samplerate
                st.info(f"🎵 Duration: {duration:.2f}s | Sample Rate: {samplerate} Hz")
                st.audio(tmp_path, format="audio/wav")
            except Exception as e:
                st.warning(f"Could not read audio properties: {e}")
                st.audio(recorded_audio_enhanced, format="audio/wav")

    except ImportError:
        st.warning("")


with tab3:
    st.markdown("**Upload a text or document file, or enter text manually**")

    text_input_method = st.radio("Choose input method:", ["✍️ Enter Text", "📄 Upload File"])
    input_text = ""

    if text_input_method == "✍️ Enter Text":
        input_text = st.text_area("Enter lyrics or text to Singify", height=200, placeholder="Type something like: 'Hello world, it's a beautiful day!'")
    else:
        uploaded_text_file = st.file_uploader(
            "Upload a text or document file",
            type=["txt", "pdf", "docx"],
            help="Supported formats: TXT, PDF, DOCX"
        )
        if uploaded_text_file:
            st.success(f"✅ Uploaded: {uploaded_text_file.name}")
            ext = uploaded_text_file.name.split(".")[-1].lower()
            try:
                if ext == "txt":
                    input_text = uploaded_text_file.read().decode("utf-8")
                elif ext == "pdf":
                    import PyPDF2
                    reader = PyPDF2.PdfReader(uploaded_text_file)
                    input_text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
                elif ext == "docx":
                    from docx import Document
                    doc = Document(uploaded_text_file)
                    input_text = "\n".join([p.text for p in doc.paragraphs])
            except Exception as e:
                st.error(f"Error reading file: {e}")

    if input_text:
        st.text_area("📝 Extracted Text", input_text, height=150)
        if st.button("🎶 Convert Text to Singify Audio", key="text_to_sing_button"):
            with st.spinner("🎤 Generating singing audio from text..."):
                try:
                    # generate PCM via synchronous helper (blocking) — acceptable inside spinner
                    pcm_data = synthesize_speech_sync(
                        f"Sing this text in a {singing_style.lower()} style with {voice_option} voice: {input_text}",
                        voice_name=voice_option
                    )
                    vocal_bytes = pcm_to_wav(pcm_data)

                    vocal_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
                    vocal_path = vocal_file.name
                    with open(vocal_path, "wb") as f:
                        f.write(vocal_bytes)

                    st.session_state.vocal_path = vocal_path
                    st.session_state.transcript = input_text
                    st.session_state.generation_complete = True
                    st.session_state.current_style = singing_style
                    st.session_state.current_voice = voice_option

                    st.success("✅ Singify generation complete!")
                    st.audio(vocal_path, format="audio/wav")

                    st.download_button(
                        "📥 Download Singified Audio",
                        vocal_bytes,
                        file_name="singified_text.wav",
                        mime="audio/wav"
                    )
                except Exception as e:
                    st.error(f"❌ Failed to generate singing voice: {e}")

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
        st.rerun()

with col2:
    if (audio_bytes or st.session_state.original_path) and st.button("ℹ️ Audio Info"):
        path_to_check = tmp_path if tmp_path else st.session_state.original_path
        if path_to_check:
            try:
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
            except Exception as e:
                st.error(f"Error reading audio info: {e}")

with col3:
    if st.session_state.generation_complete and st.button("🆕 Generate New"):
        st.session_state.transcript = None
        st.session_state.vocal_path = None
        st.session_state.generation_complete = False
        st.session_state.current_style = None
        st.session_state.current_voice = None
        st.rerun()

# -------------------------
# Step 2 & 3: Transcribe & TTS with spinner (no progress bar)
# -------------------------
async def transcribe_and_sing():
    # create client with api_key so transcription call works
    client = genai.Client(api_key=api_key)

    # Use tmp_path or stored original_path
    audio_path = tmp_path if tmp_path else st.session_state.original_path

    if not audio_path:
        st.error("No audio file available")
        return

    # Read audio for processing
    with open(audio_path, "rb") as f:
        current_audio_bytes = f.read()

    # --- Transcription ---
    with st.spinner("🔤 Transcribing..."):
        try:
            resp = client.models.generate_content(
                model="gemini-2.5-flash",
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

    # --- TTS with natural language prompt ---
    with st.spinner(f"🎵 Generating singing voice in {singing_style} style..."):
        tts_prompt = f"Sing these words in a {singing_style.lower()} style with emotion and musical expression: {st.session_state.transcript}"
        try:
            # run the synchronous synth function in a thread to avoid blocking the event loop
            loop = asyncio.get_event_loop()
            pcm_data = await loop.run_in_executor(None, lambda: synthesize_speech_sync(tts_prompt, voice_name=voice_option))
            vocal_bytes = pcm_to_wav(pcm_data)

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

            st.success("✅ Singing voice generation complete!")
        except Exception as e:
            st.error(f"❌ Generation failed: {e}")
            return

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

if audio_bytes is not None or st.session_state.original_path:
    if not st.session_state.generation_complete:
        if st.button("🎶 Transcribe & Sing", type="primary"):
            asyncio.run(transcribe_and_sing())
    else:
        st.info("✅ Generation already completed! Results shown below.")
else:
    st.warning("⚠️ Please upload or record an audio file first!")

# -------------------------
# Always Display Results if Available
# -------------------------
display_results()