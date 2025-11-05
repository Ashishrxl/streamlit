import streamlit as st
import tempfile
import numpy as np
import soundfile as sf
from pydub import AudioSegment
import matplotlib.pyplot as plt
from google import genai
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
    page_title="🎙️ AI Vocal Coach using Google Gemini",
    layout="wide"
)

# CSS styling
hide_streamlit_style = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
[data-testid="stStatusWidget"] {display: none;}
[data-testid="stToolbar"] {display: none;}
header > div:nth-child(2) { display: none; }
.main { padding: 2rem; }
.stButton > button {
    width: 100%;
    background-color: #4CAF50;
    color: white;
    padding: 0.75rem;
    font-size: 1.1rem;
}
</style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# ==============================
# Google API Key Setup
# ==============================
if "GOOGLE_API_KEY_1" not in st.secrets:
    st.error("❌ Missing GOOGLE_API_KEY_1 in Streamlit Secrets.")
    st.stop()

# ✅ new SDK: initialize client directly
client = genai.Client(api_key=st.secrets["GOOGLE_API_KEY_1"])

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
recorded_audio_native = st.audio_input("🎙️ Record your voice", key="native_recorder")

# Save uploaded/recorded audio
recorded_file_path = None
if recorded_audio_native:
    recorded_file_path = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name
    with open(recorded_file_path, "wb") as f:
        f.write(recorded_audio_native.getvalue())
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

    # --- Gemini AI Feedback (Text) ---
    st.subheader("🎶 AI Vocal Feedback")

    prompt = (
        "You are a professional vocal coach. "
        "Compare the user's singing to the reference song and provide constructive feedback "
        "about pitch accuracy, rhythm, tone, and expression. "
        "Be supportive and motivating."
    )

    with st.spinner("🎧 Analyzing vocals with Gemini..."):
        response = client.models.generate_content(
            model="gemini-2.5-pro",
            contents=[
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
            ],
        )

    st.success("✅ Feedback Ready!")
    feedback_text = response.candidates[0].content.parts[0].text
    st.write(feedback_text)

    # --- Gemini TTS Feedback ---
    st.subheader("🔊 Listen to AI Feedback")

    try:
        with st.spinner("🎙️ Generating spoken feedback..."):
            tts_response = client.models.generate_content(
                model="gemini-2.5-flash-preview-tts",
                contents=f"Speak this feedback in a warm, encouraging tone: {feedback_text}",
                config=types.GenerateContentConfig(
                    response_modalities=["AUDIO"],
                    speech_config=types.SpeechConfig(
                        voice_config=types.VoiceConfig(
                            prebuilt_voice_config=types.PrebuiltVoiceConfig(
                                voice_name="Kore"
                            )
                        )
                    )
                ),
            )

            # Extract audio bytes and save as .wav
            audio_data = tts_response.candidates[0].content.parts[0].inline_data.data
            tts_path = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name
            with wave.open(tts_path, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(24000)
                wf.writeframes(audio_data)

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