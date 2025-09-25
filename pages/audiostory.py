import streamlit as st
from google import genai
from google.genai import types
import io
import wave
import base64

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
        story = resp.text if hasattr(resp, "text") else str(resp)
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
            try:
                data = tts_resp.candidates[0].content[0].inline_data.data
            except Exception:
                data = None
            if data:
                if isinstance(data, str):
                    pcm = base64.b64decode(data)
                else:
                    pcm = bytes(data)
                wav_bytes = pcm_to_wav_bytes(pcm)
                st.audio(wav_bytes, format="audio/wav")

st.markdown("---")
st.caption("Built with Gemma + Gemini TTS")