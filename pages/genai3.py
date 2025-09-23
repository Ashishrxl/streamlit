import requests
import base64
import tempfile
import streamlit as st

# Function to synthesize speech using Gemini 2.5 Pro Preview TTS
def synthesize_speech(text, voice="en-US-Standard-B", speaking_rate=1.0):
    url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-pro-preview-tts:generateSpeech"
    headers = {
        "Authorization": f"Bearer {st.secrets['GOOGLE_API_KEY']}",
        "Content-Type": "application/json",
    }
    data = {
        "text": text,
        "audioConfig": {
            "speakingRate": speaking_rate,
            "voice": {
                "name": voice
            }
        }
    }
    response = requests.post(url, headers=headers, json=data)
    response.raise_for_status()
    audio_data = response.json().get("audio", {}).get("audioData")
    if audio_data:
        return base64.b64decode(audio_data)
    else:
        raise ValueError("No audio data received from Gemini API.")

# Step 2: Transcribe & Step 3: TTS
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
        vocal_bytes = synthesize_speech(ssml)

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