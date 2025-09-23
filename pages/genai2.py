import streamlit as st
from streamlit_audiorecorder import audio_recorder
import base64
import os
from pydub import AudioSegment
from google import genai

st.set_page_config(page_title="Singify 🎶", layout="centered")

st.title("🎤 Singify with Gemini")
st.caption("Speak a line → Transcribe with Gemini 1.5 Flash → Sing it back with Gemini 2.5 TTS")

# Sidebar
singing_style = st.sidebar.selectbox("Singing Style", ["Pop", "Ballad", "Rap", "Soft"])

# Step 1 — Record audio
st.write("### Step 1: Record your line")
audio_bytes = audio_recorder()

if audio_bytes:
    st.audio(audio_bytes, format="audio/wav")
    tmp_path = "input.wav"
    with open(tmp_path, "wb") as f:
        f.write(audio_bytes)

    if st.button("🎶 Transcribe & Sing"):
        client = genai.Client()

        # -------------------
        # 1) Transcribe with Gemini 1.5 Flash
        # -------------------
        st.info("Transcribing with Gemini 1.5 Flash...")
        with open(tmp_path, "rb") as f:
            audio_data = f.read()

        try:
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

        # -------------------
        # 2) Generate singing vocal with Gemini 2.5 Pro Preview TTS
        # -------------------
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
                voice="alloy",   # adjust if you have voice options
                format="wav"
            )
            vocal_bytes = tts_resp.get("audio") or getattr(tts_resp, "audio", None)
            if isinstance(vocal_bytes, str):
                vocal_bytes = base64.b64decode(vocal_bytes)
        except Exception as e:
            st.error(f"TTS failed: {e}")
            st.stop()

        vocal_path = "vocal.wav"
        with open(vocal_path, "wb") as f:
            f.write(vocal_bytes)

        # -------------------
        # 3) Play & download
        # -------------------
        st.audio(vocal_path, format="audio/wav")
        with open(vocal_path, "rb") as f:
            st.download_button("Download Vocal", f, file_name="singified.wav", mime="audio/wav")