import streamlit as st
import tempfile
import numpy as np
import soundfile as sf
from pydub import AudioSegment
import matplotlib.pyplot as plt
# NOTE: use google.genai client and types for the TTS path
import google.generativeai as genai
from google.genai import types
from streamlit.components.v1 import html
import wave

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

# ==============================
# Streamlit Page Setup
# ==============================
st.set_page_config(page_title="🎵 AI Vocal Coach", layout="wide")

st.title("🎙️ AI Vocal Coach using Google Gemini")
st.write("Record your voice, compare it with the reference song, and get AI-powered singing feedback!")

# ==============================
# Google API Key Setup
# ==============================
if "GOOGLE_API_KEY" not in st.secrets:
    st.error("❌ Missing GOOGLE_API_KEY in Streamlit Secrets.")
    st.stop()

# configure API key (you already used this pattern)
genai.configure(api_key=st.secrets["GOOGLE_API_KEY_1"])

# ==============================
# Helper: Load Audio + Energy Contour
# ==============================
def load_audio_energy(path):
    """Returns a normalized energy contour for visualization."""
    try:
        y, sr = sf.read(path, always_2d=False)
        if y.ndim > 1:
            y = np.mean(y, axis=1)
    except Exception:
        audio = AudioSegment.from_file(path)
        y = np.array(audio.get_array_of_samples()).astype(float)
        sr = audio.frame_rate

    frame_len = int(0.05 * sr)
    hop = int(0.025 * sr)
    energies = []

    for i in range(0, len(y) - frame_len, hop):
        frame = y[i:i + frame_len]
        energies.append(np.mean(np.abs(frame)))

    energies = np.array(energies)
    return energies / np.max(energies) if np.max(energies) != 0 else energies


# ==============================
# Step 1: Upload Reference Song
# ==============================
st.header("🎧 Step 1: Upload Reference Song")
ref_file = st.file_uploader("Upload a reference song (mp3 or wav)", type=["mp3", "wav"])

# ==============================
# Step 2: Record Singing
# ==============================
st.header("🎤 Step 2: Record Your Singing")
st.markdown("Click below to record directly from your microphone 🎙️")

# ✅ Updated Recording Section
recorded_audio_native = st.audio_input("🎙️ Record your voice", key="native_recorder")

recorded_file_path = None
if recorded_audio_native:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmpfile:
        tmpfile.write(recorded_audio_native.read())
        recorded_file_path = tmpfile.name
    st.audio(recorded_file_path, format="audio/wav")
    st.success("✅ Recording captured!")


# ==============================
# Step 3: Analyze and Get Feedback
# ==============================
if ref_file and recorded_file_path:
    ref_tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    ref_tmp.write(ref_file.read())

    # --- Energy Visualization ---
    st.subheader("📊 Comparing Energy Contours")
    ref_energy = load_audio_energy(ref_tmp.name)
    user_energy = load_audio_energy(recorded_file_path)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(ref_energy, label="Reference Song", linewidth=2)
    ax.plot(user_energy, label="Your Singing", linewidth=2)
    ax.legend()
    ax.set_title("Energy Contour Comparison")
    ax.set_xlabel("Frame")
    ax.set_ylabel("Normalized Energy")
    st.pyplot(fig)

    # --- Gemini AI Analysis ---
    st.subheader("🎶 AI Vocal Feedback")
    model = genai.GenerativeModel("models/gemini-2.5-pro")

    prompt = (
        "You are a professional vocal coach. "
        "Compare the user's singing to the reference song and provide constructive feedback "
        "about pitch accuracy, rhythm, tone, and expression. "
        "Be supportive and motivating."
    )

    with st.spinner("🎧 Analyzing vocals with Gemini..."):
        response = model.generate_content(
            [
                {
                    "role": "user",
                    "parts": [
                        {"text": prompt},
                        {
                            "inline_data": {
                                "mime_type": "audio/wav",
                                "data": open(ref_tmp.name, "rb").read(),
                            }
                        },
                        {
                            "inline_data": {
                                "mime_type": "audio/wav",
                                "data": open(recorded_file_path, "rb").read(),
                            }
                        },
                    ],
                }
            ]
        )

    st.success("✅ Feedback Ready!")
    st.write(response.text)

    # --- AI Spoken Feedback (TTS) ---
    st.subheader("🔊 Listen to AI Feedback")

    # === FIXED TTS PATH using response_modalities=["AUDIO"] and speech_config ===
    try:
        with st.spinner("🎙️ Generating spoken feedback..."):
            client = genai.Client()  # use the client interface for TTS per docs
            tts_prompt = f"Speak this feedback in a warm, encouraging tone: {response.text}"

            tts_response = client.models.generate_content(
                model="gemini-2.5-flash-preview-tts",
                contents=tts_prompt,
                config=types.GenerateContentConfig(
                    response_modalities=["AUDIO"],
                    speech_config=types.SpeechConfig(
                        voice_config=types.VoiceConfig(
                            prebuilt_voice_config=types.PrebuiltVoiceConfig(
                                voice_name="Kore"  # choose any available prebuilt voice
                            )
                        )
                    ),
                ),
            )

            # Extract PCM data (bytes) from the response (per official example)
            data = tts_response.candidates[0].content.parts[0].inline_data.data

            # Save PCM as a proper WAV file (24kHz, 16-bit, mono) so Streamlit can play it
            tts_path = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name
            with wave.open(tts_path, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)          # 16-bit = 2 bytes
                wf.setframerate(24000)     # Gemini TTS PCM uses 24000 Hz
                wf.writeframes(data)

        st.audio(tts_path, format="audio/wav")
        st.success("✅ Audio feedback generated!")
    except Exception as e:
        st.warning(f"⚠️ Audio feedback unavailable. ({e})")

    # --- Future Enhancements ---
    with st.expander("🌟 Future Enhancements"):
        st.markdown("""
        - 🎯 **Live pitch visualization** (real-time tuner)
        - 🧠 **Emotion & expression analysis**
        - 🎶 **Harmony & duet generation**
        - 🗣️ **Pronunciation feedback**
        - 📈 **Progress tracking dashboard**
        """)

else:
    st.info("Please upload a reference song and record your singing above.")