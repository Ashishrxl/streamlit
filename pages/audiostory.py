import streamlit as st
import io
import wave
import base64

from google import genai
from google.genai import types

GEMMA_MODEL = "gemma-3-12b-it"
TTS_MODEL = "gemini-2.5-flash-preview-tts"

client = genai.Client(api_key=st.secrets["GOOGLE_API_KEY"])

st.set_page_config(page_title="AI Roleplay Story", layout="wide")
st.title("AI Roleplay Story Generator")

genre = st.text_input("Enter story genre", "Cyberpunk mystery")
characters = st.text_area("List characters (comma separated)", "Detective, Hacker, AI sidekick")
length = st.selectbox("Story length", ["Short", "Medium", "Long"])
add_audio = st.checkbox("Generate audio of full story")

def pcm_to_wav_bytes(pcm_bytes, channels=1, rate=24000, sample_width=2):
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(sample_width)
        wf.setframerate(rate)
        wf.writeframes(pcm_bytes)
    buf.seek(0)
    return buf.read()

if st.button("Generate Story & Audio"):
    with st.spinner("Generating story..."):
        prompt = f"Write a {length} {genre} roleplay story with characters: {characters}. Split into scenes with dialogue."
        resp = client.models.generate_content(model=GEMMA_MODEL, contents=[prompt])
        story = getattr(resp, "text", str(resp))
        st.session_state["story"] = story

    st.subheader("Story Script")
    st.write(story)

    if add_audio:
        with st.spinner("Generating audio for full story..."):
            config = types.GenerateContentConfig(
                response_modalities=["AUDIO"],
                speech_config=types.SpeechConfig(
                    voice_config=types.VoiceConfig(
                        prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name="Kore")
                    )
                )
            )
            tts_resp = client.models.generate_content(model=TTS_MODEL, contents=[story], config=config)
            data = None
            if hasattr(tts_resp, "candidates") and tts_resp.candidates:
                candidate = tts_resp.candidates[0]
                if hasattr(candidate, "content") and candidate.content:
                    part = candidate.content[0]
                    if hasattr(part, "inline_data") and hasattr(part.inline_data, "data"):
                        data = part.inline_data.data

            if data:
                if isinstance(data, str):
                    pcm = base64.b64decode(data)
                else:
                    pcm = bytes(data)
                wav_bytes = pcm_to_wav_bytes(pcm)
                st.audio(wav_bytes, format="audio/wav")
                st.download_button(label="Download Audio", data=wav_bytes, file_name="story_audio.wav", mime="audio/wav")

st.markdown("---")
st.caption("Built with Gemma + Gemini TTS")
