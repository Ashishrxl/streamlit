import streamlit as st
import tempfile
import numpy as np
import soundfile as sf
from pydub import AudioSegment
import matplotlib.pyplot as plt
import google.generativeai as genai
from google.generativeai.types import Part
from streamlit_webrtc import webrtc_streamer, WebRtcMode, AudioProcessorBase, RTCConfiguration

# ==============================
# Streamlit Page Setup
# ==============================
st.set_page_config(page_title="🎵 AI Vocal Coach", layout="wide")

st.title("🎙️ AI Vocal Coach using Google Gemini")
st.write("Practice singing — record your voice, compare to the reference, and get AI-powered feedback!")

# ==============================
# Google API Key Setup
# ==============================
if "GOOGLE_API_KEY" not in st.secrets:
    st.error("❌ Missing GOOGLE_API_KEY in Streamlit Secrets.")
    st.stop()

genai.configure(api_key=st.secrets["GOOGLE_API_KEY_1"])

# ==============================
# Helper: Load audio + energy contour
# ==============================
def load_audio_energy(path):
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
# Section 1: Upload or Record
# ==============================
st.header("🎧 Step 1: Provide Reference Song")
ref_file = st.file_uploader("Upload a reference song (mp3 or wav)", type=["mp3", "wav"])

st.header("🎤 Step 2: Record Your Singing")
st.markdown("Click below to record directly from your microphone 🎙️")

# WebRTC configuration for mic recording
RTC_CONFIG = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

class AudioProcessor(AudioProcessorBase):
    def __init__(self):
        self.frames = []

    def recv_audio(self, frame):
        self.frames.append(frame.to_ndarray().flatten())
        return frame

webrtc_ctx = webrtc_streamer(
    key="singing-demo",
    mode=WebRtcMode.SENDONLY,
    audio_receiver_size=256,
    rtc_configuration=RTC_CONFIG,
    media_stream_constraints={"audio": True, "video": False},
    async_processing=True,
)

# Save mic recording
recorded_file_path = None
if webrtc_ctx.audio_receiver:
    audio_frames = []
    while True:
        try:
            frame = webrtc_ctx.audio_receiver.get_frame(timeout=1)
        except:
            break
        audio_frames.append(frame.to_ndarray().flatten())

    if audio_frames:
        audio_data = np.concatenate(audio_frames)
        recorded_file_path = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name
        sf.write(recorded_file_path, audio_data, 44100)
        st.audio(recorded_file_path, format="audio/wav")
        st.success("✅ Recording captured!")

# ==============================
# Section 2: Analysis & Feedback
# ==============================
if ref_file and recorded_file_path:
    ref_tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    ref_tmp.write(ref_file.read())

    # Visualization
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

    # Gemini feedback
    st.subheader("🎶 AI Feedback")
    model = genai.GenerativeModel("models/gemini-2.5-pro")

    prompt = (
        "You are a professional vocal coach. "
        "Compare the user's singing to the reference song and give detailed feedback "
        "on pitch, rhythm, and expression. Be constructive and motivating."
    )

    with st.spinner("🎧 Analyzing vocals with Gemini..."):
        response = model.generate_content([
            Part(text=prompt),
            Part(inline_data={"mime_type": "audio/wav", "data": open(ref_tmp.name, "rb").read()}),
            Part(inline_data={"mime_type": "audio/wav", "data": open(recorded_file_path, "rb").read()}),
        ])

    st.success("✅ Feedback Ready!")
    st.write(response.text)

    # Optional TTS playback
    st.subheader("🔊 AI Spoken Feedback")
    tts_model = genai.GenerativeModel("models/gemini-2.5-flash-preview-tts")
    tts_prompt = f"Speak this feedback in an encouraging tone: {response.text}"

    with st.spinner("🎙️ Generating spoken feedback..."):
        tts_response = tts_model.generate_content([Part(text=tts_prompt)])

    try:
        st.audio(tts_response.audio, format="audio/mp3")
    except Exception:
        st.warning("⚠️ Audio feedback unavailable.")

    # Future roadmap
    with st.expander("🌟 Future Enhancements"):
        st.markdown("""
        - 🎯 Real-time pitch tracking and correction visualization  
        - 🧠 Emotion and tone analysis  
        - 🎶 Harmony and background vocal generation  
        - 🗣️ Pronunciation coaching  
        - 📈 Long-term progress tracking  
        """)

else:
    st.info("Please upload a reference song and record your singing above.")