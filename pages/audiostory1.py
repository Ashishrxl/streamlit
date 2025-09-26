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

# Inputs
genre = st.text_input("Enter story genre", "Cyberpunk mystery")
characters = st.text_area("List characters (comma separated)", "Detective, Hacker, AI sidekick")
length = st.selectbox("Story length", ["Short", "Medium", "Long"])
language = st.selectbox("Select story language", ["English", "Hindi", "Bhojpuri"])

# Voice choices per language
voice_options = {
    "English": ["English Male", "English Female"],
    "Hindi": ["Hindi Male", "Hindi Female"],
    "Bhojpuri": ["Bhojpuri Male", "Bhojpuri Female"]
}
voice_choice = st.selectbox("Select voice", voice_options[language])

add_audio = st.checkbox("Generate audio of full story")

# Utility: Convert PCM → WAV
def pcm_to_wav_bytes(pcm_bytes, channels=1, rate=24000, sample_width=2):
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(sample_width)
        wf.setframerate(rate)
        wf.writeframes(pcm_bytes)
    buf.seek(0)
    return buf.read()

# Progress bar with countdown
def animate_progress_bar(progress, placeholder, text, est_time=10):
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

# Map voice choice to Gemini TTS names
def map_voice(voice_choice):
    mapping = {
        "English Male": "Kore",
        "English Female": "Charon",
        "Hindi Male": "Kore",
        "Hindi Female": "Charon",
        "Bhojpuri Male": "Kore",
        "Bhojpuri Female": "Charon"
    }
    return mapping.get(voice_choice, "Kore")

# Map language to TTS language code (Gemini can adapt pronunciation)
def map_language_code(language):
    codes = {
        "English": "en-US",
        "Hindi": "hi-IN",
        "Bhojpuri": "bh-IN"  # Gemini TTS may use hi-IN as fallback for Bhojpuri
    }
    return codes.get(language, "en-US")

# Main button: Generate story + audio
if st.button("Generate Story & Audio"):
    # --- Story Generation ---
    story_progress = st.progress(0, text="Generating story...")
    story_placeholder = st.empty()
    thread = threading.Thread(
        target=animate_progress_bar,
        args=(story_progress, story_placeholder, "Generating story", 8)
    )
    thread.start()

    # Strict prompt: only selected language + character intro + story
    prompt = (
        f"Write a {length} {genre} roleplay story in {language} ONLY. "
        f"Include first a brief introduction of each character ({characters}) and then the story. "
        f"Do NOT include any text in any other language. "
        f"Do NOT include explanations, summaries, or extra content. "
        f"The output must be entirely in {language} and contain ONLY character introductions and the story."
    )

    resp = client.models.generate_content(model=GEMMA_MODEL, contents=[prompt])
    story = getattr(resp, "text", str(resp))
    st.session_state["story"] = story

    running = False
    thread.join()
    story_progress.progress(100, text="Story generated ✅")
    story_placeholder.write("✅ Story ready!")

    st.subheader("Story Script")
    st.write(st.session_state["story"])

    # --- Audio Generation ---
    if add_audio:
        audio_progress = st.progress(0, text="Generating audio...")
        audio_placeholder = st.empty()
        thread = threading.Thread(
            target=animate_progress_bar,
            args=(audio_progress, audio_placeholder, "Generating audio", 12)
        )
        thread.start()

        config = types.GenerateContentConfig(
            response_modalities=["AUDIO"],
            speech_config=types.SpeechConfig(
                language_code=map_language_code(language),
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(
                        voice_name=map_voice(voice_choice)
                    )
                )
            )
        )
        tts_resp = client.models.generate_content(
            model=TTS_MODEL,
            contents=[st.session_state["story"]],
            config=config
        )

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
            st.session_state["audio_bytes"] = wav_bytes

# Persistent audio player + download button
if "audio_bytes" in st.session_state:
    st.audio(st.session_state["audio_bytes"], format="audio/wav")
    st.download_button(
        label="Download Audio",
        data=st.session_state["audio_bytes"],
        file_name="story_audio.wav",
        mime="audio/wav"
    )

st.markdown("---")
st.caption("Built with Gemma + Gemini TTS")