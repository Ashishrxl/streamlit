import os
import streamlit as st
from streamlit_webrtc import webrtc_streamer, AudioProcessorBase, WebRtcMode
import google.generativeai as genai
import speech_recognition as sr
import tempfile
import numpy as np
import wave
from gtts import gTTS
from langdetect import detect, DetectorFactory
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
    page_title="🎙️ TalkPlay – Voice Adventure",
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
</style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

DetectorFactory.seed = 0

st.title("🎮 TalkPlay – Voice-Controlled Adventure Game with AI Voice")

# --- Sidebar controls ---
st.header("🎧 Voice Settings")
voice_choice = st.selectbox(
    "Choose your AI narrator voice:",
    ["Narrator (English)", "Mystic (Hindi)", "Monster (Deep)", "Companion (Fast)"],
)
# Speed and pitch sliders are kept but won't affect audio for now
speed_factor = st.slider("Voice Speed (Preview Only)", 0.6, 1.5, 1.0, 0.1)
pitch_shift = st.slider("Voice Pitch (Preview Only)", -8, 8, 0, 1)

st.sidebar.markdown("---")
st.info("🎙️ Tip: Speak or type your commands below!")

# --- Configure Gemini API ---
API_KEY = st.secrets["GOOGLE_API_KEY"]
genai.configure(api_key=API_KEY)
model = genai.GenerativeModel("gemini-2.5-flash")

# --- Game state ---
GAME_DESCRIPTION = """
You are in a dark forest. Paths lead north and east.
You hear a stream nearby. Monsters might lurk around.
"""

GAME_STATE = {"location": "forest", "inventory": [], "visited": {"forest"}}

# --- Game logic ---
def process_command(cmd):
    cmd = cmd.lower()
    if "north" in cmd:
        GAME_STATE["location"] = "mountain"
        return "You walk north and find yourself at the base of a tall mountain."
    elif "east" in cmd:
        GAME_STATE["location"] = "river"
        return "You head east and reach a flowing river. You can hear water rushing."
    elif "look" in cmd:
        return f"You look around: {GAME_DESCRIPTION}"
    elif "inventory" in cmd:
        inv = ", ".join(GAME_STATE["inventory"]) or "nothing"
        return f"You are carrying {inv}."
    elif "attack" in cmd:
        return "You swing your sword into the darkness... something growls!"
    else:
        return f"The forest seems quiet... Your command '{cmd}' doesn't do much."

# --- Gemini AI narration ---
def gemini_reply(context):
    prompt = f"""
    You are TalkPlay AI, the narrator of an interactive adventure game.
    The player just said: "{context}"
    Continue the story immersively in hindi language.
    """
    response = model.generate_content(prompt)
    return response.text.strip()

# --- Text to speech (no pydub/ffmpeg) ---
def text_to_speech(text):
    try:
        lang = detect(text)
        lang_map = {"hi": "hi", "en": "en"}
        gtts_lang = lang_map.get(lang, "en")
    except Exception:
        gtts_lang = "en"

    if "Hindi" in voice_choice:
        gtts_lang = "hi"
    elif "English" in voice_choice:
        gtts_lang = "en"

    tts = gTTS(text=text, lang=gtts_lang)
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
        tts.save(f.name)
        return f.name

# --- Audio Processor (speech to text) ---
class AudioProcessor(AudioProcessorBase):
    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.text = ""

    def recv_audio(self, frame):
        audio = frame.to_ndarray().astype(np.int16)
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_wav:
                with wave.open(temp_wav.name, 'wb') as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)
                    wf.setframerate(16000)
                    wf.writeframes(audio.tobytes())
                with sr.AudioFile(temp_wav.name) as source:
                    audio_data = self.recognizer.record(source)
                    self.text = self.recognizer.recognize_google(audio_data)
        except Exception:
            pass
        return frame

# --- Chat history ---
if "history" not in st.session_state:
    st.session_state.history = [{"role": "assistant", "content": GAME_DESCRIPTION}]

for msg in st.session_state.history:
    st.chat_message(msg["role"]).markdown(msg["content"])

# --- Voice input ---
st.subheader("🎤 Speak your command below:")
webrtc_ctx = webrtc_streamer(
    key="speech",
    mode=WebRtcMode.SENDRECV,
    audio_processor_factory=AudioProcessor,
    media_stream_constraints={"audio": True, "video": False},
)

# --- Manual input ---
user_text = st.chat_input("Or type your command here...")
command = None

if webrtc_ctx and webrtc_ctx.audio_processor:
    spoken_text = webrtc_ctx.audio_processor.text
    if spoken_text:
        command = spoken_text

if user_text:
    command = user_text

if command:
    st.chat_message("user").markdown(command)
    st.session_state.history.append({"role": "user", "content": command})

    # Game logic + AI narration
    local_response = process_command(command)
    ai_text = gemini_reply(f"{command}. Context: {local_response}")
    full_response = f"{local_response}\n\n**{voice_choice} says:** {ai_text}"

    st.chat_message("assistant").markdown(full_response)
    st.session_state.history.append({"role": "assistant", "content": full_response})

    # Generate and play voice
    audio_path = text_to_speech(ai_text)
    with open(audio_path, "rb") as f:
        st.audio(f.read(), format="audio/mp3")

# --- Reset button ---
if st.button("🔄 Restart Game"):
    st.session_state.history = [{"role": "assistant", "content": GAME_DESCRIPTION}]
    GAME_STATE["location"] = "forest"
    GAME_STATE["inventory"] = []
    GAME_STATE["visited"] = {"forest"}
    st.rerun()