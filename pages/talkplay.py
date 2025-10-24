import os
import streamlit as st
from streamlit_webrtc import webrtc_streamer, AudioProcessorBase, WebRtcMode
import google.generativeai as genai
import speech_recognition as sr
import tempfile
import numpy as np
import wave
from gtts import gTTS
from streamlit.components.v1 import html


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
    page_title="My App",
    page_icon="🌐",
    initial_sidebar_state="expanded"
)


# --- CSS: Hide all unwanted items but KEEP sidebar toggle ---
hide_streamlit_style = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
[data-testid="stStatusWidget"] {display: none;}
[data-testid="stToolbar"] {display: none;}
a[href^="https://github.com"] {display: none !important;}
a[href^="https://streamlit.io"] {display: none !important;}

/* The following specifically targets and hides all child elements of the header's right side,
   while preserving the header itself and, by extension, the sidebar toggle button. */
header > div:nth-child(2) {
    display: none;
}
</style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# --- Streamlit setup ---
st.set_page_config(page_title="🎙️ TalkPlay – Voice Adventure", layout="wide")
st.title("🎮 TalkPlay – Voice-Controlled Adventure Game with AI Voice")

st.markdown("""
Speak commands like:
- “Go north”
- “Look around”
- “Check inventory”
- “Attack monster”
""")

# --- Configure Gemini API ---
API_KEY = st.secrets["GOOGLE_API_KEY"]
genai.configure(api_key=API_KEY)
model = genai.GenerativeModel("gemini-2.5-flash")

# --- Game state ---
GAME_DESCRIPTION = """
You are in a dark forest. Paths lead north and east.
You hear a stream nearby. Monsters might lurk around.
"""

GAME_STATE = {
    "location": "forest",
    "inventory": [],
    "visited": {"forest"}
}

def process_command(cmd):
    """Simple text-adventure logic for quick feedback."""
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

# --- Gemini AI Narration (text only) ---
def gemini_reply(context):
    """Get immersive narration text from Gemini."""
    prompt = f"""
    You are TalkPlay AI, the narrator of an interactive adventure game.
    The player just said: "{context}"
    Continue the story immersively and describe what happens next in a dramatic tone.
    """
    response = model.generate_content(prompt)
    return response.text.strip()

# --- Text-to-Speech (gTTS) helper ---
def text_to_speech(text):
    """Convert AI narration text to spoken audio using gTTS."""
    tts = gTTS(text)
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

# --- Chat History ---
if "history" not in st.session_state:
    st.session_state.history = [{"role": "assistant", "content": GAME_DESCRIPTION}]

for msg in st.session_state.history:
    if msg["role"] == "user":
        st.chat_message("user").markdown(msg["content"])
    else:
        st.chat_message("assistant").markdown(msg["content"])

# --- Voice input ---
st.subheader("🎤 Speak your command below:")

webrtc_ctx = webrtc_streamer(
    key="speech",
    mode=WebRtcMode.SENDRECV,
    audio_processor_factory=AudioProcessor,
    media_stream_constraints={"audio": True, "video": False},
)

# --- Manual text input ---
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
    full_response = f"{local_response}\n\n**AI Narration:** {ai_text}"

    # Display AI text
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