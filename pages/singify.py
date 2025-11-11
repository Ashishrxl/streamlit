import streamlit as st
import base64
import tempfile
import io
import asyncio
import soundfile as sf
from google import genai
import requests
import wave
import numpy as np
from streamlit.components.v1 import html

# --- Hide Streamlit branding ---
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

hide_streamlit_style = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
[data-testid="stStatusWidget"] {display: none;}
[data-testid="stToolbar"] {display: none;}
a[href^="https://github.com"] {display: none !important;}
a[href^="https://streamlit.io"] {display: none !important;}
header > div:nth-child(2) {display: none;}
</style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# --- Page setup ---
st.set_page_config(page_title="Singify 🎶", layout="centered")
st.title("🎤 Singify")
st.caption("Record or upload a line → Transcribe...")

# --- API Key selection ---
api_keys = {
    "Key 1": st.secrets["KEY_1"],
    "Key 2": st.secrets["KEY_2"],
    "Key 3": st.secrets["KEY_3"],
    "Key 4": st.secrets["KEY_4"],
    "Key 5": st.secrets["KEY_5"],
    "Key 6": st.secrets["KEY_6"],
    "Key 7": st.secrets["KEY_7"],
    "Key 8": st.secrets["KEY_8"],
    "Key 9": st.secrets["KEY_9"],
    "Key 10": st.secrets["KEY_10"],
    "Key 11": st.secrets["KEY_11"]
}
selected_key_name = st.selectbox("Select Key", list(api_keys.keys()))
api_key = api_keys[selected_key_name]

# --- Helpers ---
TOKEN_LIMIT = 8192

def estimate_tokens(text: str) -> int:
    return max(1, int(len(text) / 4))

def convert_to_wav_bytes(file_bytes):
    try:
        with io.BytesIO(file_bytes) as f:
            data, samplerate = sf.read(f, always_2d=True)
        out_bytes = io.BytesIO()
        sf.write(out_bytes, data, samplerate, format='WAV')
        return out_bytes.getvalue()
    except Exception as e:
        st.error(f"Error converting audio: {e}")
        return None

# --- Summarization ---
def summarize_text_sync(text, target_tokens=2000, model="models/gemini-2.5-pro-preview-05-06"):
    client = genai.Client(api_key=api_key)
    approx_tokens = estimate_tokens(text)
    if approx_tokens <= target_tokens:
        return text

    prompt = (
        "Summarize the following text into a concise version that preserves meaning and lyrical content. "
        f"Target approximately {target_tokens} tokens or fewer.\n\nTEXT:\n"
    )

    try:
        resp = client.models.generate_content(
            model=model,
            contents=[{"parts": [{"text": prompt + text}]}],
        )
        summary = resp.text.strip()
        if not summary:
            raise RuntimeError("Empty summary received")
        if estimate_tokens(summary) > target_tokens:
            summary = summary[:target_tokens * 4]
        return summary
    except Exception as e:
        err_msg = str(e)
        if "RESOURCE_EXHAUSTED" in err_msg or "quota" in err_msg.lower():
            st.error("🚫 API quota exceeded for this key. Please switch to another API key or wait a few minutes.")
        else:
            st.warning(f"⚠️ Summarization failed, using truncated text ({err_msg})")
        return text[:target_tokens * 4]

# --- TTS ---
def synthesize_speech_sync(text_prompt, voice_name="Kore", model_name="gemini-2.5-flash-preview-tts"):
    client = genai.Client(api_key=api_key)
    try:
        from google.genai import types  # type: ignore
    except Exception:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent"
        headers = {"x-goog-api-key": api_key, "Content-Type": "application/json"}
        data = {
            "contents": [{"parts": [{"text": text_prompt}]}],
            "generationConfig": {
                "responseModalities": ["AUDIO"],
                "speechConfig": {"voiceConfig": {"prebuiltVoiceConfig": {"voiceName": voice_name}}}
            },
            "model": model_name
        }
        try:
            resp = requests.post(url, headers=headers, json=data)
            resp.raise_for_status()
            resp_json = resp.json()
            data_field = resp_json["candidates"][0]["content"]["parts"][0]["inlineData"]["data"]
            return base64.b64decode(data_field)
        except requests.exceptions.HTTPError as e:
            err_text = str(e)
            if "RESOURCE_EXHAUSTED" in err_text or "quota" in err_text.lower():
                st.error("🚫 TTS quota exceeded. Please change the API key or retry later.")
            else:
                st.error(f"❌ TTS generation failed: {err_text}")
            return b""

    try:
        response = client.models.generate_content(
            model=model_name,
            contents=[{"parts": [{"text": text_prompt}]}],
            config=types.GenerateContentConfig(
                response_modalities=["AUDIO"],
                speech_config=types.SpeechConfig(
                    voice_config=types.VoiceConfig(
                        prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=voice_name)
                    )
                )
            )
        )
        data_field = response.candidates[0].content.parts[0].inline_data.data
        return base64.b64decode(data_field) if isinstance(data_field, str) else data_field
    except Exception as e:
        err_msg = str(e)
        if "RESOURCE_EXHAUSTED" in err_msg or "quota" in err_msg.lower():
            st.error("🚫 API quota exceeded for this key. Please switch or wait a few minutes.")
        else:
            st.error(f"❌ Failed to synthesize speech: {err_msg}")
        return b""

def pcm_to_wav(pcm_data, channels=1, sample_rate=24000, sample_width=2):
    wav_buffer = io.BytesIO()
    with wave.open(wav_buffer, 'wb') as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(sample_width)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm_data)
    return wav_buffer.getvalue()

# --- Session state ---
if 'transcript' not in st.session_state:
    st.session_state.transcript = None
if 'vocal_path' not in st.session_state:
    st.session_state.vocal_path = None
if 'original_path' not in st.session_state:
    st.session_state.original_path = None
if 'generation_complete' not in st.session_state:
    st.session_state.generation_complete = False
if 'current_style' not in st.session_state:
    st.session_state.current_style = None
if 'current_voice' not in st.session_state:
    st.session_state.current_voice = None

# --- Sidebar controls ---
singing_style = st.selectbox("Singing Style", ["Pop", "Ballad", "Rap", "Soft"])
voice_option = st.selectbox("Voice", ["Kore", "Charon", "Fenrir", "Aoede"])

# --- Audio Input ---
st.subheader("📤 Choose Audio Input Method")
tab1, tab2, tab3 = st.tabs(["📁 Upload Audio File", "🎙️ Record Audio", "📝 Upload Text or Document"])

# --- Upload Audio ---
with tab1:
    uploaded = st.file_uploader("Upload an audio file", type=["wav", "mp3", "m4a", "ogg", "flac"])
    if uploaded:
        st.success(f"✅ Uploaded: {uploaded.name} ({uploaded.size / 1024 / 1024:.2f} MB)")
        file_bytes = uploaded.read()
        ext = uploaded.name.split('.')[-1].lower()
        audio_bytes = convert_to_wav_bytes(file_bytes) if ext != "wav" else file_bytes
        if audio_bytes:
            tmp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            tmp_path = tmp_file.name
            with open(tmp_path, "wb") as f:
                f.write(audio_bytes)
            st.session_state.original_path = tmp_path
            data, samplerate = sf.read(tmp_path, always_2d=True)
            duration = len(data) / samplerate
            st.info(f"🎵 Duration: {duration:.2f}s | Sample Rate: {samplerate} Hz | Channels: {data.shape[1]}")
            st.audio(tmp_path, format="audio/wav")

# --- Record Audio ---
with tab2:
    st.write("🎙️ Record audio directly in your browser (coming soon).")

# --- Upload Text ---
with tab3:
    st.write("Upload a text file or paste lyrics manually.")
    text_input = st.text_area("✍️ Enter Lyrics or Text", height=200)
    text_file = st.file_uploader("📄 Or upload a text document", type=["txt", "pdf", "docx"])
    text_data = None

    if text_file:
        try:
            if text_file.type == "application/pdf":
                from PyPDF2 import PdfReader
                reader = PdfReader(text_file)
                text_data = "\n".join(page.extract_text() for page in reader.pages)
            elif text_file.type in ["application/vnd.openxmlformats-officedocument.wordprocessingml.document"]:
                from docx import Document
                doc = Document(text_file)
                text_data = "\n".join([p.text for p in doc.paragraphs])
            else:
                text_data = text_file.read().decode("utf-8")
            st.success("✅ Text extracted successfully.")
        except Exception as e:
            st.error(f"Failed to read text file: {e}")

    if st.button("🎶 Generate Singing Voice"):
        if not text_input and not text_data:
            st.error("Please enter or upload text first.")
        else:
            text_to_use = text_input or text_data
            st.info("Summarizing text (if too long)...")
            summarized = summarize_text_sync(text_to_use)
            st.info("Synthesizing singing voice...")
            pcm_audio = synthesize_speech_sync(summarized, voice_name=voice_option)
            if pcm_audio:
                wav_data = pcm_to_wav(pcm_audio)
                st.audio(wav_data, format="audio/wav")
                st.download_button("💾 Download Audio", wav_data, file_name="singify_output.wav")