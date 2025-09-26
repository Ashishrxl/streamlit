import streamlit as st
import io
import wave
import base64
import time
import threading

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
language = st.selectbox("Select story language", ["English", "Hindi", "Bhojpuri"])
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

def animate_progress_bar(progress, placeholder, text, est_time=10):
    """Animate progress bar with countdown until stopped by global flag."""
    global running
    running = True
    start = time.time()
    while running:
        elapsed = time.time() - start
        remaining = max(0, est_time - int(elapsed))
        pct = min(100, int((elapsed / est_time) * 100))
        progress.progress(pct, text=f"{text} ~{remaining}s left")
        placeholder.write(f"⏳ {text} (about {remaining}s remaining)")
        time.sleep(0.2)
        if pct >= 100:
            break

if st.button("Generate Story & Audio"):
    # Story progress bar + countdown
    story_progress = st.progress(0, text="Generating story...")
    story_placeholder = st.empty()
    thread = threading.Thread(target=animate_progress_bar, args=(story_progress, story_placeholder, "Generating story", 8))
    thread.start()

    prompt = (
        f"Write a {length} {genre} roleplay story in {language} "
        f"with characters: {characters}. Split into scenes with dialogue."
    )
    resp = client.models.generate_content(model=GEMMA_MODEL, contents=[prompt])
    story = getattr(resp, "text", str(resp))
    st.session_state["story"] = story

    running = False
    thread.join()
    story_progress.progress(100, text="Story generated ✅")
    story_placeholder.write("✅ Story ready!")

    st.subheader("Story Script")
    st.write(story)

    if add_audio:
        # Audio progress bar + countdown
        audio_progress = st.progress(0, text="Generating audio...")
        audio_placeholder = st.empty()
        thread = threading.Thread(target=animate_progress_bar, args=(audio_progress, audio_placeholder, "Generating audio", 12))
        thread.start()

        config = types.GenerateContentConfig(
            response_modalities=["AUDIO"],
            speech_config=types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    # Using the same voice, but Gemini adapts output to language in text
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name="Kore")
                )
            )
        )
        tts_resp = client.models.generate_content(model=TTS_MODEL, contents=[story], config=config)

        data = None
        if hasattr(tts_resp, "candidates") and tts_resp.candidates:
            candidate = tts_resp.candidates[0]
            if hasattr(candidate, "content") and hasattr(candidate.content, "parts"):
                for part in candidate.content.parts:
                    if hasattr(part, "inline_data") and hasattr(part.inline_data, "data"):
                        data = part.inline_data.data
                        break

        running = False
        thread.join()
        audio_progress.progress(100, text="Audio generated ✅")
        audio_placeholder.write("✅ Audio ready!")

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