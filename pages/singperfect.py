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
import base64
import os
from contextlib import contextmanager

# ==============================
# Hide Streamlit default elements
# ==============================
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

st.set_page_config(
    page_title="🎙️ AI Vocal Coach using Google Gemini",
    layout="wide"
)

# ==============================
# Helper utilities
# ==============================
@contextmanager
def temp_wav_file(suffix=".wav"):
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.close()
    try:
        yield tmp.name
    finally:
        try:
            os.remove(tmp.name)
        except Exception:
            pass


def safe_read_audio(path):
    try:
        y, sr = sf.read(path, always_2d=False)
        if y.ndim > 1:
            y = np.mean(y, axis=1)
        return y.astype(float), sr
    except Exception:
        audio = AudioSegment.from_file(path)
        y = np.array(audio.get_array_of_samples()).astype(float)
        sr = audio.frame_rate
        return y, sr


@st.cache_data(show_spinner=False)
def load_audio_energy(path):
    y, sr = safe_read_audio(path)
    frame_len = int(0.05 * sr)
    hop = int(0.025 * sr)
    energies = []

    if len(y) < frame_len:
        return np.array([0.0])

    for i in range(0, len(y) - frame_len, hop):
        frame = y[i:i + frame_len]
        energies.append(np.mean(np.abs(frame)))

    energies = np.array(energies)
    max_e = np.max(energies) if energies.size else 0.0
    return energies / max_e if max_e != 0 else energies


def write_bytes_to_wav(path, audio_bytes, nchannels=1, sampwidth=2, framerate=24000):
    try:
        with wave.open(path, "wb") as wf:
            wf.setnchannels(nchannels)
            wf.setsampwidth(sampwidth)
            wf.setframerate(framerate)
            wf.writeframes(audio_bytes)
        return True
    except Exception:
        return False


# ==============================
# Gemini client (cached)
# ==============================
@st.cache_resource
def get_gemini_client():
    if "GOOGLE_API_KEY_1" not in st.secrets:
        return None
    return genai.Client(api_key=st.secrets["GOOGLE_API_KEY_1"])


# ==============================
# Step 1: Choose feedback options
# ==============================
st.header("⚙️ Step 1: Choose Feedback Options")

col1, col2 = st.columns(2)
with col1:
    feedback_lang = st.selectbox("🗣️ Choose feedback language", ["English", "Hindi"])

with col2:
    enable_audio_feedback = st.checkbox("🔊 Generate Audio Feedback (optional)", value=False)

voice_choice = st.selectbox("🎤 Choose AI voice (for TTS)", ["Kore", "Ava", "Wave"], index=0)

# ==============================
# Step 2: Upload Reference Song
# ==============================
st.header("🎧 Step 2: Upload Reference Song")
ref_file = st.file_uploader("Upload a reference song (mp3 or wav)", type=["mp3", "wav"])

MAX_MB = 50
lyrics_text = None

if ref_file is not None and hasattr(ref_file, "size"):
    size_mb = ref_file.size / (1024 * 1024)
    if size_mb > MAX_MB:
        st.warning(f"⚠️ File size {size_mb:.1f} MB may take longer to process.")

    # --- Gemini Lyrics Extraction ---
    client = get_gemini_client()
    if client is None:
        st.error("❌ Missing GOOGLE_API_KEY_1 in Streamlit Secrets.")
        st.stop()

    ref_tmp_for_lyrics = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    ref_tmp_for_lyrics.write(ref_file.read())
    ref_tmp_for_lyrics.flush()

    st.subheader("📝 Extracted Lyrics from Song")
    with st.spinner("🎵 Extracting lyrics using Gemini..."):
        try:
            lyrics_response = client.models.generate_content(
                model="gemini-2.5-pro",
                contents=[
                    {
                        "role": "user",
                        "parts": [
                            {
                                "text": (
                                    "Listen to this song and transcribe or extract its lyrics as accurately as possible. "
                                    "If lyrics are unclear, infer reasonable text but avoid guessing nonsensical words."
                                )
                            },
                            {
                                "inline_data": {
                                    "mime_type": "audio/wav",
                                    "data": open(ref_tmp_for_lyrics.name, "rb").read(),
                                }
                            },
                        ],
                    }
                ],
            )
            try:
                lyrics_text = lyrics_response.candidates[0].content.parts[0].text
            except Exception:
                lyrics_text = getattr(lyrics_response.candidates[0].content, "text", "") or "(No lyrics detected.)"
        except Exception as e:
            st.warning(f"⚠️ Could not extract lyrics automatically. ({e})")
            lyrics_text = "(Lyrics unavailable.)"

# ==============================
# Step 3: Record Singing (Side-by-side with Lyrics)
# ==============================
st.header("🎤 Step 3: Record Your Singing")

col_rec, col_lyrics = st.columns([1, 1])
with col_rec:
    recorded_audio_native = st.audio_input("🎙️ Record your voice", key="native_recorder")

    recorded_file_path = None
    if recorded_audio_native:
        recorded_file_path = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name
        with open(recorded_file_path, "wb") as f:
            f.write(recorded_audio_native.getvalue())
        st.success("✅ Recording captured!")

with col_lyrics:
    if lyrics_text:
        st.markdown("### 🎶 Lyrics to Sing Along")
        st.markdown(
            f"""
            <div id="lyrics-box" style="
                height: 350px;
                overflow-y: auto;
                background-color: #f9f9f9;
                padding: 1rem;
                border-radius: 12px;
                font-size: 1.05rem;
                line-height: 1.6;
                white-space: pre-wrap;
            ">{lyrics_text}</div>
            <script>
                let box = document.getElementById('lyrics-box');
                let scrollPos = 0;
                setInterval(() => {{
                    if (box.scrollTop < box.scrollHeight - box.clientHeight) {{
                        scrollPos += 1;
                        box.scrollTop = scrollPos;
                    }}
                }}, 120);
            </script>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.info("Lyrics will appear here after you upload a song.")

# ==============================
# Step 4: Analyze and Get Feedback
# ==============================
client = get_gemini_client()
if client is None:
    st.error("❌ Missing GOOGLE_API_KEY_1 in Streamlit Secrets.")
    st.stop()

if ref_file and recorded_audio_native:
    ref_tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    try:
        ref_tmp.write(ref_file.read())
        ref_tmp.flush()
        ref_tmp_name = ref_tmp.name

        # --- Energy Visualization ---
        st.subheader("📊 Comparing Energy Contours")
        with st.spinner("🔎 Computing energy contours..."):
            ref_energy = load_audio_energy(ref_tmp_name)
            user_energy = load_audio_energy(recorded_file_path)

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(ref_energy, label="Reference Song", linewidth=2)
        ax.plot(user_energy, label="Your Singing", linewidth=2)
        ax.legend()
        ax.set_title("Energy Contour Comparison")
        ax.set_xlabel("Frame")
        ax.set_ylabel("Normalized Energy")
        st.pyplot(fig)

        # Side-by-side players
        st.markdown("**🎧 Listen to both recordings:**")
        c1, c2 = st.columns(2)
        with c1:
            st.audio(ref_tmp_name)
            st.caption("Reference Song")
        with c2:
            st.audio(recorded_file_path)
            st.caption("Your Recording")

        # --- Gemini AI Feedback ---
        st.subheader("🎶 AI Vocal Feedback")

        lang_instruction = (
            "Provide feedback in English."
            if feedback_lang == "English"
            else "Provide feedback in Hindi using natural, encouraging tone."
        )

        prompt = (
            f"You are a professional vocal coach. Compare the user's singing to the reference song "
            f"and give supportive feedback about pitch, rhythm, tone, and expression. {lang_instruction}"
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
                                    "data": open(ref_tmp_name, "rb").read(),
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
        try:
            feedback_text = response.candidates[0].content.parts[0].text
        except Exception:
            feedback_text = getattr(response.candidates[0].content, "text", "") or "(No feedback returned.)"

        st.write(feedback_text)

        # --- Gemini TTS Feedback (Optional) ---
        if enable_audio_feedback:
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
                                        voice_name=voice_choice
                                    )
                                )
                            )
                        ),
                    )

                    audio_part = None
                    try:
                        audio_part = tts_response.candidates[0].content.parts[0]
                    except Exception:
                        audio_part = getattr(tts_response.candidates[0].content, "parts", [None])[0]

                    if audio_part is None or not hasattr(audio_part, "inline_data"):
                        raise ValueError("TTS response did not contain audio data")

                    audio_data = audio_part.inline_data.data
                    tts_path = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name
                    success_write = write_bytes_to_wav(tts_path, audio_data)
                    if not success_write:
                        with open(tts_path, "wb") as f:
                            f.write(audio_data)

                st.audio(tts_path, format="audio/wav")
                st.success("✅ Audio feedback generated!")

                with open(tts_path, "rb") as f:
                    audio_bytes = f.read()
                    b64 = base64.b64encode(audio_bytes).decode()
                    href = f'<a href="data:audio/wav;base64,{b64}" download="AI_Vocal_Feedback.wav">🎵 Download AI Feedback Audio</a>'
                    st.markdown(href, unsafe_allow_html=True)

            except Exception as e:
                st.warning(f"⚠️ Audio feedback unavailable. ({e})")

    finally:
        try:
            ref_tmp.close()
        except Exception:
            pass
else:
    st.info("Please upload a reference song and record your singing above.")