import streamlit as st
import base64
from google import genai
import tempfile

st.set_page_config(page_title="Singify 🎶", layout="centered")
st.title("🎤 Singify with Gemini")
st.caption("Record or upload a line → Transcribe with Gemini 1.5 Flash → Sing it back with Gemini 2.5 TTS")

# Sidebar: Singing style
singing_style = st.sidebar.selectbox("Singing Style", ["Pop", "Ballad", "Rap", "Soft"])

# -------------------------
# Step 1: Record or Upload Audio
# -------------------------
st.write("### Step 1: Record or Upload your voice (WAV/MP3/M4A)")

audio_bytes = None
tmp_path = None

# --- Upload file ---
uploaded = st.file_uploader("Upload your audio file", type=["wav", "mp3", "m4a"])
if uploaded:
    audio_bytes = uploaded.read()
    st.audio(audio_bytes, format="audio/wav")
    tmp_path = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
    with open(tmp_path, "wb") as f:
        f.write(audio_bytes)

# --- Record using browser ---
if st.button("🎙 Record Voice (browser)"):
    st.info("Recording feature not natively available without extra components. Please upload an audio file for now.")

# -------------------------
# Step 2: Transcribe & Step 3: TTS
# -------------------------
if audio_bytes and st.button("🎶 Transcribe & Sing"):
    client = genai.Client()
    st.info("Transcribing with Gemini 1.5 Flash...")

    try:
        with open(tmp_path, "rb") as f:
            audio_data = f.read()

        resp = client.models.generate_content(
            model="gemini-1.5-flash",
            contents=[
                {"role": "user", "parts": [
                    {"text": "Please transcribe this speech."},
                    {"inline_data": {"mime_type": "audio/wav", "data": base64.b64encode(audio_data).decode()}}
                ]}
            ]
        )

        transcript = resp.text
        st.success("Transcribed text:")
        st.write(transcript)

    except Exception as e:
        st.error(f"Transcription failed: {e}")
        st.stop()

    st.info("Generating singing-style voice with Gemini 2.5 Pro Preview TTS...")

    ssml = f"""
    <speak>
      <prosody rate="95%" pitch="+2st">
        Sing these words in a {singing_style} style: {transcript}
      </prosody>
    </speak>
    """

    try:
        tts_resp = client.audio.synthesize(
            model="gemini-2.5-pro-preview-tts",
            input=ssml,
            voice="alloy",
            format="wav"
        )

        vocal_bytes = tts_resp.get("audio") or getattr(tts_resp, "audio", None)
        if isinstance(vocal_bytes, str):
            vocal_bytes = base64.b64decode(vocal_bytes)

    except Exception as e:
        st.error(f"TTS failed: {e}")
        st.stop()

    # Save vocal
    vocal_path = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
    with open(vocal_path, "wb") as f:
        f.write(vocal_bytes)

    # Play & download
    st.success("Here is your sung version:")
    st.audio(vocal_path, format="audio/wav")

    with open(vocal_path, "rb") as f:
        st.download_button("Download Vocal", f, file_name="singified.wav", mime="audio/wav")