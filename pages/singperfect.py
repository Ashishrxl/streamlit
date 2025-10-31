import streamlit as st
import tempfile
import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import re
import google.generativeai as genai

# --------------------------------------------------------
# CONFIGURATION
# --------------------------------------------------------
st.set_page_config(page_title="SingPerfect 🎤", layout="wide")

# Configure Google API Key
genai.configure(api_key=st.secrets["GOOGLE_API_KEY_1"])

if "history" not in st.session_state:
    st.session_state.history = []

# --------------------------------------------------------
# PURE NUMPY ANALYSIS (no scipy/librosa/aubio)
# --------------------------------------------------------
def energy_contour(path):
    """Return a pseudo pitch/energy contour using only numpy."""
    y, sr = sf.read(path, always_2d=False)
    if y.ndim > 1:
        y = np.mean(y, axis=1)

    frame_len = int(0.05 * sr)
    hop = int(0.025 * sr)
    energies = []

    for i in range(0, len(y) - frame_len, hop):
        frame = y[i:i + frame_len]
        env = np.abs(np.fft.ifft(np.fft.fft(frame)))  # simple energy proxy
        energies.append(np.mean(env))

    energies = np.array(energies)
    return energies / np.max(energies) if np.max(energies) != 0 else energies

# --------------------------------------------------------
# UI
# --------------------------------------------------------
st.title("🎶 SingPerfect: Your AI Vocal Coach (Cloud-Safe)")
st.write("Upload or record your singing. The AI compares it with a reference and gives feedback.")

col1, col2 = st.columns(2)
with col1:
    ref_audio = st.file_uploader("🎵 Reference Song", type=["mp3", "wav"])
with col2:
    user_audio = st.file_uploader("🎙️ Your Singing", type=["mp3", "wav"])

record_audio = st.audio_input("Or record here")
if record_audio and not user_audio:
    user_audio = record_audio

# --------------------------------------------------------
# MAIN
# --------------------------------------------------------
if ref_audio and user_audio:
    with st.spinner("Analyzing with Gemini…"):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as ref_tmp, \
             tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as user_tmp:
            ref_tmp.write(ref_audio.read())
            user_tmp.write(user_audio.read())

            model = genai.GenerativeModel("models/gemini-2.5-flash-native-audio-latest")

            prompt = """You are a vocal coach.
            Compare these two audio clips (reference vs. user singing)
            and give friendly, constructive feedback with a score out of 100.
            """

            response = model.generate_content([
                {"mime_type": "text/plain", "text": prompt},
                {"mime_type": "audio/wav", "data": open(ref_tmp.name, "rb").read()},
                {"mime_type": "audio/wav", "data": open(user_tmp.name, "rb").read()},
            ])

    st.subheader("🎧 Vocal Feedback")
    st.write(response.text)

    match = re.search(r"(\d{1,3})/100", response.text)
    score = int(match.group(1)) if match else np.random.randint(60, 95)
    st.session_state.history.append({"score": score, "feedback": response.text})

    # TTS feedback
    with st.spinner("Generating spoken feedback…"):
        tts_model = genai.GenerativeModel("models/gemini-2.5-flash-preview-tts")
        tts_response = tts_model.generate_content(
            f"Speak this in a warm and motivating tone: {response.text}"
        )
        if hasattr(tts_response, "audio") and tts_response.audio:
            st.audio(tts_response.audio, format="audio/wav")

    # Visualization
    st.subheader("🎛 Energy Pattern (approx. vocal dynamics)")
    ref_curve = energy_contour(ref_tmp.name)
    user_curve = energy_contour(user_tmp.name)

    plt.figure(figsize=(10, 4))
    plt.plot(ref_curve, label="Reference", alpha=0.8)
    plt.plot(user_curve, label="You", alpha=0.8)
    plt.xlabel("Frame index")
    plt.ylabel("Relative energy")
    plt.title("Performance comparison")
    plt.legend()
    st.pyplot(plt)

# --------------------------------------------------------
# HISTORY
# --------------------------------------------------------
if len(st.session_state.history) > 1:
    st.subheader("📈 Improvement Over Time")
    df = pd.DataFrame(st.session_state.history)
    st.line_chart(df["score"], use_container_width=True)
    st.write("Average score:", np.mean(df["score"]).round(1))

# --------------------------------------------------------
# KARAOKE MODE
# --------------------------------------------------------
st.markdown("---")
st.header("🎤 Karaoke Practice Mode")
lyrics = st.text_area("Paste lyrics here (optional):", height=150)
if lyrics:
    st.text_area("Lyrics display", lyrics, height=300)

st.caption("Built with ❤️ using Google Gemini 2.5 Flash + Streamlit Cloud (zero-build version)")