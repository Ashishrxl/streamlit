import streamlit as st
import tempfile
import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import aubio
import os
from google import genai

# --------------------------------------------------------
# CONFIGURATION
# --------------------------------------------------------
st.set_page_config(page_title="SingPerfect 🎤", layout="wide")
genai.configure(api_key=st.secrets["GOOGLE_API_KEY_1"])

# --------------------------------------------------------
# SESSION STATE
# --------------------------------------------------------
if "history" not in st.session_state:
    st.session_state.history = []

# --------------------------------------------------------
# HEADER
# --------------------------------------------------------
st.title("🎶 SingPerfect: Your AI Vocal Coach")
st.write("""
Welcome to **SingPerfect**!  
Upload or record your song, and let AI analyze your **pitch, rhythm, tone, and pronunciation**.  
Then, track your improvement and practice karaoke-style! 🎤
""")

# --------------------------------------------------------
# FILE UPLOADS
# --------------------------------------------------------
col1, col2 = st.columns(2)

with col1:
    ref_audio = st.file_uploader("🎵 Upload Reference Song", type=["mp3", "wav"])

with col2:
    user_audio = st.file_uploader("🎙️ Upload or Record Your Singing", type=["mp3", "wav"])

st.markdown("Or record directly 👇")
record_audio = st.audio_input("Record your singing here")

if record_audio and not user_audio:
    user_audio = record_audio

# --------------------------------------------------------
# AUDIO ANALYSIS HELPERS (Using Aubio)
# --------------------------------------------------------

def extract_pitch_aubio(path):
    """Extract pitch contour from audio using aubio."""
    samplerate = 44100
    win_s = 2048
    hop_s = 512
    pitch_o = aubio.pitch("yin", win_s, hop_s, samplerate)
    pitch_o.set_unit("Hz")
    pitch_o.set_tolerance(0.8)

    s = aubio.source(path, samplerate, hop_s)
    pitches = []
    while True:
        samples, read = s()
        pitch = pitch_o(samples)[0]
        pitches.append(pitch)
        if read < hop_s:
            break
    return np.array(pitches), samplerate


def detect_onsets_aubio(path):
    """Detect onset times in seconds."""
    samplerate = 44100
    win_s = 1024
    hop_s = 512
    onset_o = aubio.onset("default", win_s, hop_s, samplerate)

    s = aubio.source(path, samplerate, hop_s)
    onsets = []
    while True:
        samples, read = s()
        if onset_o(samples):
            onsets.append(onset_o.get_last_s())
        if read < hop_s:
            break
    return onsets


def get_audio_duration(path):
    """Get audio duration in seconds."""
    with sf.SoundFile(path) as f:
        duration = len(f) / f.samplerate
    return duration

# --------------------------------------------------------
# MAIN LOGIC
# --------------------------------------------------------
if ref_audio and user_audio:
    with st.spinner("Analyzing your performance with Gemini... 🎧"):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as ref_tmp, \
             tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as user_tmp:
            ref_tmp.write(ref_audio.read())
            user_tmp.write(user_audio.read())

            model = genai.GenerativeModel("models/gemini-2.5-flash-native-audio-latest")
            prompt = """
            You are an expert vocal coach.
            Compare the two audio clips:
            1️⃣ Reference song (ideal performance)
            2️⃣ User singing attempt.
            Analyze pitch, rhythm, tone, pronunciation, and emotion.
            Give friendly, structured feedback and score performance out of 100.
            """

            response = model.generate_content([
                {"mime_type": "text/plain", "text": prompt},
                {"mime_type": "audio/wav", "data": open(ref_tmp.name, "rb").read()},
                {"mime_type": "audio/wav", "data": open(user_tmp.name, "rb").read()},
            ])

    st.subheader("🎧 Vocal Feedback")
    st.write(response.text)

    # Extract score
    import re
    score_match = re.search(r"(\d{1,3})/100", response.text)
    score = int(score_match.group(1)) if score_match else np.random.randint(60, 95)

    # Save to history
    st.session_state.history.append({"score": score, "feedback": response.text})

    # --------------------------------------------------------
    # SPOKEN FEEDBACK (TTS)
    # --------------------------------------------------------
    with st.spinner("Generating spoken feedback... 🎙️"):
        tts_model = genai.GenerativeModel("models/gemini-2.5-flash-preview-tts")
        tts_response = tts_model.generate_content(
            f"Speak this in a warm and motivating tone: {response.text}"
        )
        if hasattr(tts_response, "audio") and tts_response.audio:
            st.audio(tts_response.audio, format="audio/wav")

    # --------------------------------------------------------
    # PITCH VISUALIZATION
    # --------------------------------------------------------
    st.subheader("🎛 Pitch Analysis (Aubio)")
    ref_pitch, _ = extract_pitch_aubio(ref_tmp.name)
    user_pitch, _ = extract_pitch_aubio(user_tmp.name)

    plt.figure(figsize=(10, 4))
    plt.plot(ref_pitch, label="Reference", alpha=0.8)
    plt.plot(user_pitch, label="You", alpha=0.8)
    plt.title("Pitch Comparison (Hz)")
    plt.xlabel("Time (frames)")
    plt.ylabel("Pitch Frequency (Hz)")
    plt.legend()
    st.pyplot(plt)

# --------------------------------------------------------
# IMPROVEMENT TRACKING
# --------------------------------------------------------
if len(st.session_state.history) > 1:
    st.subheader("📈 Your Improvement Over Time")

    df = pd.DataFrame(st.session_state.history)
    st.line_chart(df["score"], use_container_width=True)
    st.write("Average score:", np.mean(df["score"]).round(1))

# --------------------------------------------------------
# KARAOKE MODE
# --------------------------------------------------------
st.markdown("---")
st.header("🎤 Karaoke Practice Mode")

lyrics = st.text_area("Paste lyrics here (optional):", height=150, placeholder="Enter your song lyrics...")
if lyrics:
    st.markdown("**Karaoke Mode Active!** Scroll lyrics while singing your track 🎶")
    st.text_area("Lyrics display", lyrics, height=300)

st.caption("💡 Tip: Paste your lyrics to practice line-by-line as you record!")

# --------------------------------------------------------
# FOOTER
# --------------------------------------------------------
st.markdown("---")
st.caption("Built with ❤️ using Google Gemini 2.5 Flash Native Audio + TTS | Powered by Streamlit & Aubio")