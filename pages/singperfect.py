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
import shutil

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
# Check ffmpeg availability
# ==============================
if not shutil.which("ffmpeg"):
    st.warning("⚠️ ffmpeg not found — please ensure ffmpeg is installed for MP3 decoding.")

# ==============================
# Helper utilities
# ==============================
@contextmanager
def temp_wav_file(suffix=".wav"):
    """Context manager for a temporary WAV file."""
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
    """Read audio robustly with soundfile or fallback to pydub."""
    try:
        y, sr = sf.read(path, always_2d=False)
        if y.ndim > 1:
            y = np.mean(y, axis=1)
        return y.astype(float), sr
    except Exception:
        try:
            audio = AudioSegment.from_file(path)
            y = np.array(audio.get_array_of_samples()).astype(float)
            sr = audio.frame_rate
            return y, sr
        except Exception as e:
            st.error(f"❌ Could not decode audio file: {e}")
            raise e


@st.cache_data(show_spinner=False)
def load_audio_energy(path):
    """Returns normalized energy contour for visualization."""
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
    """Write PCM bytes to WAV file safely."""
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
if ref_file is not None and hasattr(ref_file, "size"):
    size_mb = ref_file.size / (1024 * 1024)
    if size_mb > MAX_MB:
        st.warning(f"⚠️ File size {size_mb:.1f} MB may take longer to process.")

lyrics_text = ""

client = get_gemini_client()
if client is None:
    st.error("❌ Missing GOOGLE_API_KEY_1 in Streamlit Secrets.")
    st.stop()

if ref_file:
    st.subheader("📝 Extracting Lyrics")
    with st.spinner("🎶 Extracting lyrics from the song using Gemini..."):
        ref_tmp_path = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name
        ref_bytes = ref_file.read()
        with open(ref_tmp_path, "wb") as f:
            f.write(ref_bytes)
        try:
            response = client.models.generate_content(
                model="gemini-2.5-pro",
                contents=[
                    {
                        "role": "user",
                        "parts": [
                            {"text": "Extract the complete lyrics (if any) from this audio and return only the lyrics, no commentary."},
                            {
                                "inline_data": {
                                    "mime_type": "audio/wav",
                                    "data": open(ref_tmp_path, "rb").read(),
                                }
                            },
                        ],
                    }
                ],
            )
            lyrics_text = response.candidates[0].content.parts[0].text.strip()
        except Exception:
            lyrics_text = "Lyrics could not be extracted."

# ==============================
# Step 3: Record Singing
# ==============================
st.header("🎤 Step 3: Record Your Singing")
recorded_audio_native = st.audio_input("🎙️ Record your voice", key="native_recorder")

recorded_file_path = None
if recorded_audio_native:
    recorded_file_path = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name
    with open(recorded_file_path, "wb") as f:
        f.write(recorded_audio_native.getvalue())
    st.success("✅ Recording captured!")

# ==============================
# Show Lyrics (Karaoke View)
# ==============================
if ref_file and lyrics_text:
    st.header("📜 Lyrics (Sing Along)")

    col_audio, col_lyrics = st.columns([1, 1.5])

    with col_audio:
        st.audio(ref_tmp_path, format="audio/wav")
        st.caption("🎧 Reference Song")

    with col_lyrics:
        try:
            audio = AudioSegment.from_file(ref_tmp_path)
            duration_sec = audio.duration_seconds
        except Exception:
            duration_sec = 60

        lines = [line.strip() for line in lyrics_text.strip().split("\n") if line.strip()]
        if len(lines) == 0:
            st.info("Lyrics unavailable or empty.")
        else:
            n_lines = len(lines)
            timestamps = [round(i * (duration_sec / n_lines), 2) for i in range(n_lines)]
            lines_html = "".join(
                [f'<p class="lyric-line" data-time="{timestamps[i]}">{lines[i]}</p>' for i in range(n_lines)]
            )
            karaoke_html = f"""
            <div>
                <audio id="karaokePlayer" controls style="width:100%;">
                    <source src="data:audio/wav;base64,{base64.b64encode(open(ref_tmp_path,"rb").read()).decode()}" type="audio/wav">
                    Your browser does not support the audio element.
                </audio>
                <div id="lyrics-box" style="height:350px;overflow-y:auto;border:1px solid #ccc;
                    padding:10px;font-size:1.1rem;line-height:1.6;margin-top:10px;">
                    {lines_html}
                </div>
            </div>
            <script>
            const audio = document.getElementById('karaokePlayer');
            const lines = Array.from(document.querySelectorAll('.lyric-line'));
            const times = lines.map(l => parseFloat(l.dataset.time));
            let active = 0;
            function highlight(time) {{
                for (let i=0;i<lines.length;i++) {{
                    if (time >= times[i] && (i===lines.length-1 || time < times[i+1])) {{
                        if (active!==i) {{
                            lines.forEach(l=>l.style.color='#444');
                            lines[i].style.color='#ff4081';
                            lines[i].scrollIntoView({{behavior:'smooth', block:'center'}});
                            active=i;
                        }}
                        break;
                    }}
                }}
            }}
            audio.addEventListener('timeupdate',()=>highlight(audio.currentTime));
            audio.addEventListener('seeked',()=>highlight(audio.currentTime));
            </script>
            """
            html(karaoke_html, height=420)

# ==============================
# Step 4: Analyze and Get Feedback
# ==============================
if ref_file and recorded_file_path:
    ref_tmp_name = ref_tmp_path
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

                audio_part = tts_response.candidates[0].content.parts[0]
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

else:
    st.info("Please upload a reference song and record your singing above.")