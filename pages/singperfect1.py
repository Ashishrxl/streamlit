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
import threading
import queue
import sounddevice as sd

# ==============================
# Hide Streamlit elements
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

st.markdown("""
<style>
footer {pointer-events:none;}
#MainMenu {visibility:hidden;}
footer {visibility:hidden;}
[data-testid="stStatusWidget"], [data-testid="stToolbar"] {display:none;}
.main {padding:2rem;}
.stButton>button {
    width:100%;
    background:#4CAF50;
    color:white;
    padding:0.75rem;
    font-size:1.1rem;
}
</style>
""", unsafe_allow_html=True)

st.set_page_config(page_title="🎙️ AI Vocal Coach", layout="wide")

# ==============================
# Utility functions
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
    """Try reading audio robustly."""
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
    for i in range(0, len(y) - frame_len, hop):
        frame = y[i:i + frame_len]
        energies.append(np.mean(np.abs(frame)))
    energies = np.array(energies)
    if np.max(energies) > 0:
        energies /= np.max(energies)
    return energies

# ==============================
# Gemini client
# ==============================
@st.cache_resource
def get_gemini_client():
    if "GOOGLE_API_KEY_1" not in st.secrets:
        return None
    return genai.Client(api_key=st.secrets["GOOGLE_API_KEY_1"])

client = get_gemini_client()
if client is None:
    st.error("❌ Missing GOOGLE_API_KEY in secrets.")
    st.stop()

# ==============================
# Mode Selection
# ==============================
st.title("🎙️ AI Vocal Coach")

mode = st.radio("Choose Mode:", ["Upload & Compare", "🎧 Live Coaching"])

# =======================================================
#  MODE 1: UPLOAD & COMPARE (original functionality)
# =======================================================
if mode == "Upload & Compare":
    st.header("⚙️ Step 1: Choose Feedback Options")
    col1, col2 = st.columns(2)
    with col1:
        feedback_lang = st.selectbox("🗣️ Feedback language", ["English", "Hindi"])
    with col2:
        enable_audio_feedback = st.checkbox("🔊 Generate Audio Feedback", value=False)

    voice_choice = st.selectbox("🎤 Choose AI voice", ["Kore", "Ava", "Wave"], index=0)

    # Step 2: Upload Song
    st.header("🎧 Step 2: Upload Reference Song")
    ref_file = st.file_uploader("Upload a song (mp3 or wav)", type=["mp3", "wav"])

    if "lyrics_text" not in st.session_state:
        st.session_state.lyrics_text = ""
    if "ref_tmp_path" not in st.session_state:
        st.session_state.ref_tmp_path = None

    if ref_file and not st.session_state.lyrics_text:
        with st.spinner("🎵 Extracting lyrics..."):
            tmp_path = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name
            with open(tmp_path, "wb") as f:
                f.write(ref_file.read())
            st.session_state.ref_tmp_path = tmp_path
            try:
                response = client.models.generate_content(
                    model="gemini-2.5-flash-live",
                    contents=[
                        {"role": "user", "parts": [
                            {"text": "Extract the complete lyrics from this song and return only the text."},
                            {"inline_data": {"mime_type": "audio/wav", "data": open(tmp_path, "rb").read()}}
                        ]}
                    ]
                )
                st.session_state.lyrics_text = response.candidates[0].content.parts[0].text.strip()
            except Exception:
                st.session_state.lyrics_text = "Lyrics could not be extracted."

    if st.session_state.ref_tmp_path and st.session_state.lyrics_text:
        st.subheader("📜 Lyrics (Sing Along)")
        lines = [line.strip() for line in st.session_state.lyrics_text.split("\n") if line.strip()]
        try:
            audio = AudioSegment.from_file(st.session_state.ref_tmp_path)
            duration = audio.duration_seconds
        except Exception:
            duration = 60
        timestamps = [round(i * (duration / len(lines)), 2) for i in range(len(lines))]
        lines_html = "".join([f'<p class="lyric-line" data-time="{timestamps[i]}">{lines[i]}</p>' for i in range(len(lines))])
        karaoke_html = f"""
        <div>
          <audio id="karaokePlayer" controls style="width:100%;">
            <source src="data:audio/wav;base64,{base64.b64encode(open(st.session_state.ref_tmp_path,'rb').read()).decode()}" type="audio/wav">
          </audio>
          <div id="lyrics-box" style="height:350px;overflow-y:auto;border:1px solid #ccc;
            padding:10px;font-size:1.1rem;line-height:1.6;margin-top:10px;">{lines_html}</div>
        </div>
        <script>
        const audio=document.getElementById('karaokePlayer');
        const lines=Array.from(document.querySelectorAll('.lyric-line'));
        const times=lines.map(l=>parseFloat(l.dataset.time));
        let active=0;
        function highlight(time){{
            for(let i=0;i<lines.length;i++){{
                if(time>=times[i]&&(i===lines.length-1||time<times[i+1])){{
                    if(active!==i){{
                        lines.forEach(l=>l.style.color='#444');
                        lines[i].style.color='#ff4081';
                        lines[i].scrollIntoView({{behavior:'smooth',block:'center'}});
                        active=i;
                    }}
                    break;
                }}
            }}
        }}
        audio.addEventListener('timeupdate',()=>highlight(audio.currentTime));
        </script>
        """
        html(karaoke_html, height=420)

    # Step 3: Record
    st.header("🎤 Step 3: Record Your Singing")
    recorded_audio_native = st.audio_input("🎙️ Record your voice", key="recorder")

    recorded_file_path = None
    if recorded_audio_native:
        recorded_file_path = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name
        with open(recorded_file_path, "wb") as f:
            f.write(recorded_audio_native.getvalue())
        st.success("✅ Recording captured!")

    # Step 4: Compare + Feedback
    if st.session_state.ref_tmp_path and recorded_file_path:
        st.subheader("🎶 Reference vs Your Singing")
        col_a, col_b = st.columns(2)
        with col_a:
            st.audio(st.session_state.ref_tmp_path)
            st.caption("🎧 Reference Song")
        with col_b:
            st.audio(recorded_file_path)
            st.caption("🎤 Your Recording")

        with st.spinner("🔍 Analyzing energy patterns..."):
            ref_energy = load_audio_energy(st.session_state.ref_tmp_path)
            user_energy = load_audio_energy(recorded_file_path)

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(ref_energy, label="Reference", linewidth=2)
        ax.plot(user_energy, label="You", linewidth=2)
        ax.legend()
        ax.set_title("Energy Contour Comparison")
        st.pyplot(fig)

        st.subheader("💬 AI Vocal Feedback")

        lang_instruction = (
            "Provide feedback in English."
            if feedback_lang == "English"
            else "Provide feedback in Hindi using a natural tone."
        )

        prompt = f"You are a professional vocal coach. Compare the user's singing to the reference and give supportive feedback about pitch, rhythm, tone, and expression. {lang_instruction}"

        with st.spinner("🎧 Generating feedback..."):
            response = client.models.generate_content(
                model="gemini-2.5-flash-live",
                contents=[
                    {"role": "user", "parts": [
                        {"text": prompt},
                        {"inline_data": {"mime_type": "audio/wav", "data": open(st.session_state.ref_tmp_path, "rb").read()}},
                        {"inline_data": {"mime_type": "audio/wav", "data": open(recorded_file_path, "rb").read()}},
                    ]}
                ]
            )

        try:
            feedback_text = response.candidates[0].content.parts[0].text
        except Exception:
            feedback_text = "No feedback generated."

        st.write(feedback_text)

        if enable_audio_feedback:
            with st.spinner("🔊 Generating spoken feedback..."):
                try:
                    tts = client.models.generate_content(
                        model="gemini-2.5-flash-live",
                        contents=f"Speak this feedback warmly: {feedback_text}",
                        config=types.GenerateContentConfig(
                            response_modalities=["AUDIO"],
                            speech_config=types.SpeechConfig(
                                voice_config=types.VoiceConfig(
                                    prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=voice_choice)
                                )
                            )
                        ),
                    )
                    audio_part = tts.candidates[0].content.parts[0]
                    audio_data = audio_part.inline_data.data
                    tts_path = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name
                    with open(tts_path, "wb") as f:
                        f.write(audio_data)
                    st.audio(tts_path)
                    st.success("✅ Audio feedback ready!")
                except Exception as e:
                    st.warning(f"⚠️ Audio feedback failed: {e}")
    else:
        st.info("Please upload a song and record your voice to continue.")

# =======================================================
#  MODE 2: REALTIME COACHING
# =======================================================
else:
    st.header("🎧 Live Coaching Mode (Real-Time AI)")

    q = queue.Queue()

    def stream_microphone_audio(session):
        def callback(indata, frames, time, status):
            q.put(bytes(indata))
        with sd.RawInputStream(samplerate=16000, blocksize=1024, dtype="int16", channels=1, callback=callback):
            st.info("🎙️ Listening... (Press Stop below when done)")
            while True:
                data = q.get()
                if data == b"STOP":
                    break
                session.send_input({"audio": data})

    def receive_responses(session):
        st.markdown("**AI Feedback (Live):**")
        for event in session.receive():
            if event.type == "response.output_text.delta":
                st.write(event.data, end="")
            elif event.type == "response.output_audio.delta":
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
                    f.write(event.data)
                    st.audio(f.name, format="audio/wav")

    if st.button("🎤 Start Live Coaching"):
        session = client.live.start(
            model="gemini-2.5-flash-native-audio-dialog",
            config=types.LiveConfig(
                response_modalities=["AUDIO", "TEXT"],
                speech_config=types.SpeechConfig(
                    voice_config=types.VoiceConfig(
                        prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name="Ava")
                    )
                ),
            ),
        )

        sender = threading.Thread(target=stream_microphone_audio, args=(session,))
        receiver = threading.Thread(target=receive_responses, args=(session,))

        sender.start()
        receiver.start()

        if st.button("🛑 Stop"):
            q.put(b"STOP")
            session.close()
            st.success("✅ Session ended.")