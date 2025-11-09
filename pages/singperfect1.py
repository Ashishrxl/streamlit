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
import time
import json
import websocket

# ============================
# UI: hide Streamlit chrome
# ============================
st.set_page_config(page_title="🎙️ AI Vocal Coach", layout="wide")
html(
    """
    <script>
    try {
      const sel = window.top.document.querySelectorAll('[href*="streamlit.io"], [href*="streamlit.app"]');
      sel.forEach(e => e.style.display='none');
    } catch(e) { console.warn('parent DOM not reachable', e); }
    </script>
    """,
    height=0,
)

st.markdown(
    """
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
    font-size:1.05rem;
}
</style>
""",
    unsafe_allow_html=True,
)

# ============================
# Utilities
# ============================
@contextmanager
def temp_wav_file(suffix=".wav"):
    f = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    f.close()
    try:
        yield f.name
    finally:
        try:
            os.remove(f.name)
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
    for i in range(0, len(y) - frame_len, hop):
        frame = y[i:i + frame_len]
        energies.append(np.mean(np.abs(frame)))
    energies = np.array(energies)
    if energies.size == 0:
        return energies
    if np.max(energies) > 0:
        energies /= np.max(energies)
    return energies

# ============================
# Gemini client
# ============================
@st.cache_resource
def get_gemini_client():
    if "GOOGLE_API_KEY" not in st.secrets:
        return None
    return genai.Client(api_key=st.secrets["GOOGLE_API_KEY"])

client = get_gemini_client()
if client is None:
    st.error("❌ Missing GOOGLE_API_KEY in Streamlit secrets.")
    st.stop()

API_KEY = st.secrets["GOOGLE_API_KEY"]

# ============================
# WebSocket endpoint (correct v1beta bidi)
# ============================
REALTIME_URL = (
    "wss://generativelanguage.googleapis.com/ws/"
    "google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent"
    f"?key={API_KEY}"
)

# ============================
# Tabs: Upload & Compare | Live
# ============================
tab1, tab2 = st.tabs(["🎧 Upload & Compare", "🎤 Live Coaching (Realtime)"])

# ----------------------------
# TAB 1: Upload & Compare
# ----------------------------
with tab1:
    st.header("⚙️ Step 1: Choose Feedback Options")
    col1, col2 = st.columns(2)
    with col1:
        feedback_lang = st.selectbox("🗣️ Feedback language", ["English", "Hindi"])
    with col2:
        enable_audio_feedback_upload = st.checkbox("🔊 Generate Audio Feedback (for upload mode)", value=False)
    voice_choice_upload = st.selectbox("🎤 Choose AI voice (upload mode)", ["Kore", "Ava", "Wave"], index=0)

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
                    model="gemini-2.5-pro",
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

    # show lyrics + karaoke
    if st.session_state.ref_tmp_path and st.session_state.lyrics_text:
        st.subheader("📜 Lyrics (Sing Along)")
        lines = [line.strip() for line in st.session_state.lyrics_text.split("\n") if line.strip()]
        try:
            audio = AudioSegment.from_file(st.session_state.ref_tmp_path)
            duration = audio.duration_seconds
        except Exception:
            duration = 60
        timestamps = [round(i * (duration / max(len(lines), 1)), 2) for i in range(len(lines))]
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

    # recording in upload flow
    st.header("🎤 Step 3: Record Your Singing")
    recorded_audio_native = st.audio_input("🎙️ Record your voice (upload mode)", key="recorder_upload")
    recorded_file_path = None
    if recorded_audio_native:
        recorded_file_path = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name
        with open(recorded_file_path, "wb") as f:
            f.write(recorded_audio_native.getvalue())
        st.success("✅ Recording captured!")

    # compare + analysis
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
        if ref_energy.size:
            ax.plot(ref_energy, label="Reference", linewidth=2)
        if user_energy.size:
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
            try:
                response = client.models.generate_content(
                    model="gemini-2.5-pro",
                    contents=[
                        {"role": "user", "parts": [
                            {"text": prompt},
                            {"inline_data": {"mime_type": "audio/wav", "data": open(st.session_state.ref_tmp_path, "rb").read()}},
                            {"inline_data": {"mime_type": "audio/wav", "data": open(recorded_file_path, "rb").read()}},
                        ]}
                    ]
                )
                feedback_text = response.candidates[0].content.parts[0].text
            except Exception as e:
                feedback_text = f"No feedback generated ({e})"

        st.write(feedback_text)

        if enable_audio_feedback_upload:
            with st.spinner("🔊 Generating spoken feedback..."):
                try:
                    tts = client.models.generate_content(
                        model="gemini-2.5-flash-preview-tts",
                        contents=f"Speak this feedback warmly: {feedback_text}",
                        config=types.GenerateContentConfig(
                            response_modalities=["AUDIO"],
                            speech_config=types.SpeechConfig(
                                voice_config=types.VoiceConfig(
                                    prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=voice_choice_upload)
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
        st.info("Please upload a song and record your voice to continue (upload mode).")

# ----------------------------
# TAB 2: Live Coaching (Realtime)
# ----------------------------
with tab2:
    st.header("🎤 Live Coaching (Gemini Realtime)")
    st.markdown("Stream your voice live and get instant spoken feedback — session is recorded and analyzed after you stop.")

    # live-mode options
    colx1, colx2 = st.columns(2)
    with colx1:
        enable_audio_feedback_live = st.checkbox("🔊 Generate Audio Summary After Session", value=True)
    with colx2:
        voice_choice_live = st.selectbox("🎤 AI voice for summary", ["Kore", "Ava", "Wave"], index=1)

    # persistent state for live session
    if "live_running" not in st.session_state:
        st.session_state.live_running = False
    if "recorded_frames" not in st.session_state:
        st.session_state.recorded_frames = []  # list of np arrays (int16)
    if "live_logs" not in st.session_state:
        st.session_state.live_logs = []

    status_placeholder = st.empty()
    log_placeholder = st.empty()
    control_col1, control_col2 = st.columns([1, 1])

    def push_log(msg):
        st.session_state.live_logs.append(str(msg))
        # keep last 50
        st.session_state.live_logs = st.session_state.live_logs[-50:]

    def websocket_send_setup(ws):
        setup_msg = {
            "setup": {
                "model": "gemini-2.5-flash-native-audio-dialog",
                "generation_config": {
                    "response_modalities": ["AUDIO", "TEXT"],
                    "speech_config": {
                        "voice_config": {"prebuilt_voice_config": {"voice_name": voice_choice_live}}
                    },
                },
            }
        }
        ws.send(json.dumps(setup_msg))

    def mic_and_send(ws, samplerate):
        """Raw microphone streaming callback; appends frames to session_state and sends base64 to ws."""
        def callback(indata, frames, time_info, status):
            if not st.session_state.live_running:
                raise Exception("stop")
            # ensure int16
            arr = indata.copy()
            if arr.dtype != np.int16:
                # convert float to int16 if needed
                if np.issubdtype(arr.dtype, np.floating):
                    arr = (arr * 32767).astype(np.int16)
                else:
                    arr = arr.astype(np.int16)
            st.session_state.recorded_frames.append(arr)
            try:
                ws.send(json.dumps({"input_audio_buffer": {"data": base64.b64encode(arr.tobytes()).decode("utf-8")}}))
            except Exception as e:
                push_log(f"⚠️ send error: {e}")

        try:
            import sounddevice as sd_local
            with sd_local.RawInputStream(samplerate=samplerate, blocksize=1024, dtype="int16", channels=1, callback=callback):
                while st.session_state.live_running:
                    time.sleep(0.05)
            # notify completion
            try:
                ws.send(json.dumps({"input_audio_buffer": {"complete": True}}))
            except Exception:
                pass
        except Exception as e:
            push_log(f"⚠️ Mic stream error: {e}")
            st.session_state.live_running = False

    def receive_loop(ws):
        """Receive loop from websocket; push textual logs and save any returned audio to temp files and show them."""
        try:
            while st.session_state.live_running:
                msg = ws.recv()
                if not msg:
                    continue
                try:
                    data = json.loads(msg)
                except Exception:
                    continue
                # Some responses include data under data/parts
                if data.get("response") and data["response"].get("output"):
                    for part in data["response"]["output"]:
                        if "text" in part:
                            push_log("AI: " + part["text"])
                        # audio returned as base64 in 'data'
                        if "data" in part:
                            try:
                                audio_bytes = base64.b64decode(part["data"])
                                tmp_audio = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
                                tmp_audio.write(audio_bytes)
                                tmp_audio.close()
                                push_log(f"[AI audio saved: {tmp_audio.name}]")
                                # display small player in logs area by storing marker; UI will read these markers and show players
                                push_log(("__AUDIO__", tmp_audio.name))
                            except Exception as e:
                                push_log(f"⚠️ Unable to save AI audio: {e}")
        except Exception:
            # socket closed or thread ended
            pass

    def start_live():
        """Start websocket connection and mic streaming in background threads."""
        push_log("Attempting to connect to Gemini Realtime...")
        st.session_state.recorded_frames = []
        st.session_state.live_logs = []
        try:
            ws = websocket.WebSocket()
            ws.connect(REALTIME_URL)
        except Exception as e:
            push_log(f"❌ Connection failed: {e}")
            st.session_state.live_running = False
            return

        try:
            websocket_send_setup(ws)
            push_log("✅ Connected to Gemini Realtime (start singing).")
        except Exception as e:
            push_log(f"❌ Setup failed: {e}")
            try:
                ws.close()
            except Exception:
                pass
            st.session_state.live_running = False
            return

        # determine samplerate
        try:
            samplerate = int(np.round(sf.SoundFile.__dict__.get('samplerate', 16000)))
        except Exception:
            samplerate = 16000
        # better: try to query default device
        try:
            import sounddevice as sd_local
            dev = sd_local.query_devices(kind="input")
            samplerate = int(dev.get("default_samplerate", samplerate))
        except Exception:
            samplerate = 16000

        # set running flag
        st.session_state.live_running = True

        # start threads
        threading.Thread(target=mic_and_send, args=(ws, samplerate), daemon=True).start()
        threading.Thread(target=receive_loop, args=(ws,), daemon=True).start()

    def stop_and_analyze():
        """Stop live, save recording, then analyze with Gemini generate_content and optionally TTS audio summary."""
        st.session_state.live_running = False
        push_log("🛑 Stopping session...")
        # create WAV from recorded_frames (list of np arrays)
        if not st.session_state.recorded_frames:
            push_log("⚠️ No recorded frames to save.")
            return

        try:
            all_arr = np.concatenate(st.session_state.recorded_frames, axis=0)
            # ensure int16
            if all_arr.dtype != np.int16:
                if np.issubdtype(all_arr.dtype, np.floating):
                    all_arr = (all_arr * 32767).astype(np.int16)
                else:
                    all_arr = all_arr.astype(np.int16)
            out_path = os.path.abspath("my_live_session.wav")
            sf.write(out_path, all_arr, 16000, subtype='PCM_16')
            push_log(f"✅ Saved recording: {out_path}")
        except Exception as e:
            push_log(f"⚠️ Could not save recording: {e}")
            return

        # Play the saved session in UI by adding special marker to logs
        push_log(("__AUDIO__", out_path))

        # Now analyze with Gemini (generate_content) — give supportive coaching feedback
        with st.spinner("🔍 Analyzing your performance and generating feedback..."):
            prompt = (
                "You are a professional vocal coach. Listen to the provided audio and give a supportive, "
                "actionable critique about pitch (flat/sharp), rhythm (timing), tone (brightness/warmth), "
                "breath control, and emotional expression. Provide specific short exercises the singer can try."
            )
            try:
                with open(out_path, "rb") as af:
                    response = client.models.generate_content(
                        model="gemini-2.5-pro",
                        contents=[
                            {"role": "user", "parts": [
                                {"text": prompt},
                                {"inline_data": {"mime_type": "audio/wav", "data": af.read()}}
                            ]}
                        ],
                    )
                feedback_text = response.candidates[0].content.parts[0].text
                push_log("=== AI Vocal Feedback ===")
                push_log(feedback_text)
            except Exception as e:
                push_log(f"⚠️ Analysis failed: {e}")
                return

        # Optionally produce a spoken summary via TTS
        if enable_audio_feedback_live:
            try:
                tts_prompt = f"Speak this feedback warmly and clearly: {feedback_text}"
                tts_resp = client.models.generate_content(
                    model="gemini-2.5-flash-preview-tts",
                    contents=tts_prompt,
                    config=types.GenerateContentConfig(
                        response_modalities=["AUDIO"],
                        speech_config=types.SpeechConfig(
                            voice_config=types.VoiceConfig(
                                prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=voice_choice_live)
                            )
                        )
                    )
                )
                audio_part = tts_resp.candidates[0].content.parts[0]
                audio_bytes = audio_part.inline_data.data
                tts_path = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
                with open(tts_path.name, "wb") as wf:
                    wf.write(audio_bytes)
                push_log(("__AUDIO__", tts_path.name))
                push_log("✅ Spoken feedback ready.")
            except Exception as e:
                push_log(f"⚠️ TTS failed: {e}")

    # Buttons and UI control
    if not st.session_state.live_running:
        if control_col1.button("🚀 Start Live Coaching"):
            # check mic presence
            try:
                import sounddevice as sd_check
                sd_check.query_devices(kind="input")
            except Exception as e:
                st.warning("⚠️ No microphone detected or permission denied. Live coaching won't start.")
                push_log(f"⚠️ Mic check failed: {e}")
            else:
                st.session_state.live_logs = []
                start_live()
    else:
        if control_col2.button("🛑 Stop & Save Session"):
            stop_and_analyze()

    # show status & logs; also display any audio markers as players
    status_placeholder.markdown(f"**Live running:** {st.session_state.live_running}")
    # render logs with audio playback for markers
    logs = st.session_state.live_logs.copy()
    if logs:
        for entry in logs:
            if isinstance(entry, tuple) and entry[0] == "__AUDIO__":
                try:
                    st.audio(entry[1], format="audio/wav")
                except Exception:
                    st.markdown(f"🔊 Audio saved: `{entry[1]}`")
            else:
                st.markdown(f"- {entry}")