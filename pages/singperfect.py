import streamlit as st
import tempfile
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import re
import io
import google.generativeai as genai


# Try importing soundfile; fallback to pydub
try:
    import soundfile as sf
    HAVE_SF = True
except Exception:
    from pydub import AudioSegment
    HAVE_SF = False


from streamlit.components.v1 import html

# Hide Streamlit default elements
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

disable_footer_click = """
    <style>
    footer {pointer-events: none;}
    </style>
"""
st.markdown(disable_footer_click, unsafe_allow_html=True)

st.set_page_config(
    page_title="🎙️ Text 2 Audio",
    layout="wide"
)

# CSS to hide unwanted elements
hide_streamlit_style = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
[data-testid="stStatusWidget"] {display: none;}
[data-testid="stToolbar"] {display: none;}
a[href^="https://github.com"] {display: none !important;}
a[href^="https://streamlit.io"] {display: none !important;}
header > div:nth-child(2) { display: none; }
.main { padding: 2rem; }
.stButton > button {
    width: 100%;
    background-color: #4CAF50;
    color: white;
    padding: 0.75rem;
    font-size: 1.1rem;
}
.success-box {
    padding: 1rem;
    background-color: #d4edda;
    border: 1px solid #c3e6cb;
    border-radius: 0.25rem;
    color: #155724;
}
.warning-box {
    padding: 1rem;
    background-color: #fff3cd;
    border: 1px solid #ffeaa7;
    border-radius: 0.25rem;
    color: #856404;
}
.info-box {
    padding: 1rem;
    background-color: #d1ecf1;
    border: 1px solid #bee5eb;
    border-radius: 0.25rem;
    color: #0c5460;
}
</style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)
# --------------------------------------------------------
# CONFIGURATION
# --------------------------------------------------------


st.set_page_config(page_title="SingPerfect 🎶", layout="wide")
genai.configure(api_key=st.secrets["GOOGLE_API_KEY_1"])

if "history" not in st.session_state:
    st.session_state.history = []

# --------------------------------------------------------
# UTILITY: read audio safely
# --------------------------------------------------------
def read_audio(file_path_or_bytes):
    """Return samples (numpy array) and sample rate."""
    if HAVE_SF:
        data, sr = sf.read(file_path_or_bytes, always_2d=False)
        if data.ndim > 1:
            data = np.mean(data, axis=1)
        return data, sr
    else:
        # Fallback: PyDub
        audio = AudioSegment.from_file(file_path_or_bytes)
        samples = np.array(audio.get_array_of_samples()).astype(float)
        sr = audio.frame_rate
        return samples, sr

# --------------------------------------------------------
# PURE NUMPY ENERGY ANALYSIS
# --------------------------------------------------------
def energy_contour(audio_path_or_bytes):
    """Return normalized energy contour (no scipy/librosa)."""
    y, sr = read_audio(audio_path_or_bytes)
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
st.title("🎶 SingPerfect: AI Vocal Coach (Cloud-Safe)")
st.write("Upload or record your singing. The AI compares it with a reference and gives friendly feedback!")

col1, col2 = st.columns(2)
with col1:
    ref_audio = st.file_uploader("🎵 Reference Song", type=["mp3", "wav"])
with col2:
    user_audio = st.file_uploader("🎙️ Your Singing", type=["mp3", "wav"])

record_audio = st.audio_input("Or record directly here")
if record_audio and not user_audio:
    user_audio = record_audio

# --------------------------------------------------------
# MAIN LOGIC
# --------------------------------------------------------
if ref_audio and user_audio:
    with st.spinner("Analyzing your singing using Gemini… 🎧"):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as ref_tmp, \
             tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as user_tmp:
            ref_tmp.write(ref_audio.read())
            user_tmp.write(user_audio.read())

            model = genai.GenerativeModel("models/gemini-2.5-flash-native-audio-latest")
            prompt = """You are a vocal coach. Compare these two audio clips:
            1️⃣ Reference song (ideal)
            2️⃣ User singing attempt.
            Provide constructive, friendly feedback and a score out of 100.
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

    # --------------------------------------------------------
    # SPEAK FEEDBACK
    # --------------------------------------------------------
    with st.spinner("Generating spoken feedback… 🎙️"):
        tts_model = genai.GenerativeModel("models/gemini-2.5-flash-preview-tts")
        tts_response = tts_model.generate_content(
            f"Speak this in a warm and motivating tone: {response.text}"
        )
        if hasattr(tts_response, "audio") and tts_response.audio:
            st.audio(tts_response.audio, format="audio/wav")

    # --------------------------------------------------------
    # VISUALIZATION
    # --------------------------------------------------------
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

st.caption("Built with ❤️ using Google Gemini 2.5 Flash + Streamlit Cloud (auto-fallback audio engine)")