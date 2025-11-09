import streamlit as st import tempfile import numpy as np import soundfile as sf from pydub import AudioSegment import matplotlib.pyplot as plt from google import genai from google.genai import types from streamlit.components.v1 import html import wave import base64 import os from contextlib import contextmanager

-----------------------------

UI tweaks: hide default Streamlit elements

-----------------------------

html( """ <script> try { const sel = window.top.document.querySelectorAll('[href*="streamlit.io"], [href*="streamlit.app"]'); sel.forEach(e => e.style.display='none'); } catch(e) { console.warn('parent DOM not reachable', e); } </script> """, height=0 )

disable_footer_click = """ <style> footer {pointer-events: none;} </style> """ st.markdown(disable_footer_click, unsafe_allow_html=True)

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
</style>""" st.markdown(hide_streamlit_style, unsafe_allow_html=True)

st.set_page_config( page_title="🎙️ AI Vocal Coach using Google Gemini", layout="wide" )

-----------------------------

Helper utilities

-----------------------------

@contextmanager def temp_wav_file(suffix=".wav"): """Context manager that yields a temp filename and ensures cleanup.""" tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix) tmp.close() try: yield tmp.name finally: try: os.remove(tmp.name) except Exception: pass

def safe_read_audio(path): """Read audio robustly with soundfile, fallback to pydub. Returns mono numpy array and sr.""" try: y, sr = sf.read(path, always_2d=False) if y.ndim > 1: y = np.mean(y, axis=1) return y.astype(float), sr except Exception: audio = AudioSegment.from_file(path) y = np.array(audio.get_array_of_samples()).astype(float) sr = audio.frame_rate # pydub may give stereo interleaved arrays; already converted above return y, sr

@st.cache_data(show_spinner=False) def load_audio_energy(path): """Returns a normalized energy contour for visualization. Cached for same files.""" y, sr = safe_read_audio(path)

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

def write_bytes_to_wav(path, audio_bytes, nchannels=1, sampwidth=2, framerate=24000): """Write raw PCM bytes to a WAV file safely.""" try: with wave.open(path, "wb") as wf: wf.setnchannels(nchannels) wf.setsampwidth(sampwidth) wf.setframerate(framerate) wf.writeframes(audio_bytes) return True except Exception: return False

-----------------------------

Gemini client: cached resource

-----------------------------

@st.cache_resource def get_gemini_client(): if "GOOGLE_API_KEY_1" not in st.secrets: return None return genai.Client(api_key=st.secrets["GOOGLE_API_KEY_1"])

-----------------------------

UI: Step 1 - options (preserve original options)

-----------------------------

st.header("⚙️ Step 1: Choose Feedback Options") col1, col2 = st.columns(2)

with col1: feedback_lang = st.selectbox("🗣️ Choose feedback language", ["English", "Hindi"])

with col2: enable_audio_feedback = st.checkbox("🔊 Generate Audio Feedback (optional)", value=False)

Add voice choice but do not remove original behavior; default matches original voice name

voice_choice = st.selectbox("🎤 Choose AI voice (affects TTS)", ["Kore", "Ava", "Wave"], index=0)

-----------------------------

Step 2: Upload reference song (preserve uploader)

-----------------------------

st.header("🎧 Step 2: Upload Reference Song") ref_file = st.file_uploader("Upload a reference song (mp3 or wav)", type=["mp3", "wav"])

file size limit (informational but does not block) - keeps original behavior but warns

MAX_MB = 50 if ref_file is not None and hasattr(ref_file, "size"): size_mb = ref_file.size / (1024 * 1024) if size_mb > MAX_MB: st.warning(f"Uploaded reference file is {size_mb:.1f} MB. Large files may take longer to process.")

-----------------------------

Step 3: Record Singing (preserve original st.audio_input usage)

-----------------------------

st.header("🎤 Step 3: Record Your Singing") recorded_audio_native = st.audio_input("🎙️ Record your voice", key="native_recorder")

Save recorded audio to temp file (preserve logic but add validation and cleanup)

recorded_file_path = None if recorded_audio_native: recorded_file_path = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name with open(recorded_file_path, "wb") as f: f.write(recorded_audio_native.getvalue()) st.audio(recorded_file_path, format="audio/wav") st.success("✅ Recording captured!")

-----------------------------

Step 4: Analyze and Get Feedback (preserve core flow + improve reliability)

-----------------------------

client = get_gemini_client()

if client is None: st.error("❌ Missing or invalid GOOGLE_API_KEY_1 in Streamlit Secrets. Please add it to continue.")

if ref_file and recorded_file_path and client is not None: # Keep original tempfile behavior but use context managers where possible ref_tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav") try: ref_tmp.write(ref_file.read()) ref_tmp.flush() ref_tmp_name = ref_tmp.name

# --- Energy Visualization --- (preserve plotting but improve UX)
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

    # Side-by-side audio players (UX improvement, does not remove anything)
    st.markdown("**Listen to both recordings**")
    pcol1, pcol2 = st.columns(2)
    with pcol1:
        st.audio(ref_tmp_name)
        st.caption("Reference song")
    with pcol2:
        st.audio(recorded_file_path)
        st.caption("Your recording")

    # --- Gemini AI Feedback (Text) ---
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

    # Gemini call wrapped in try/except and with small UX progress
    try:
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
        # The original code expected content.parts[0].text — keep same extraction but check safely
        try:
            feedback_text = response.candidates[0].content.parts[0].text
        except Exception:
            # Fallback: join candidate texts if structure differs
            feedback_text = getattr(response.candidates[0].content, "text", "") or ""

        if not feedback_text:
            feedback_text = "(No textual feedback returned by the model.)"

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

                    # Try to extract audio bytes robustly
                    audio_part = None
                    try:
                        audio_part = tts_response.candidates[0].content.parts[0]
                    except Exception:
                        # fallback: maybe content is different shape
                        audio_part = getattr(tts_response.candidates[0].content, "parts", [None])[0]

                    if audio_part is None or not hasattr(audio_part, "inline_data"):
                        raise ValueError("TTS response did not contain audio inline_data")

                    audio_data = audio_part.inline_data.data

                    # Determine sample rate and channels from metadata if present (best-effort)
                    # Fallback to original values used in earlier code
                    tts_path = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name
                    success_write = write_bytes_to_wav(tts_path, audio_data)

                    if not success_write:
                        # As a fallback, attempt to write raw bytes directly
                        with open(tts_path, "wb") as f:
                            f.write(audio_data)

                # Play and provide download link
                st.audio(tts_path, format="audio/wav")
                st.success("✅ Audio feedback generated!")

                with open(tts_path, "rb") as f:
                    audio_bytes = f.read()
                    b64 = base64.b64encode(audio_bytes).decode()
                    href = f'<a href="data:audio/wav;base64,{b64}" download="AI_Vocal_Feedback.wav">🎵 Download AI Feedback Audio</a>'
                    st.markdown(href, unsafe_allow_html=True)

            except Exception as e:
                st.warning(f"⚠️ Audio feedback unavailable. ({e})")

    except Exception as e:
        st.error(f"Failed to call Gemini API: {e}")

finally:
    try:
        ref_tmp.close()
    except Exception:
        pass
    # Do not remove recorded_file_path here — preserve original files during session

else: st.info("Please upload a reference song and record your singing above.")

