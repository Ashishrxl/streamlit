import os
import streamlit as st
from streamlit_webrtc import webrtc_streamer, AudioProcessorBase, WebRtcMode
import google.generativeai as genai
import speech_recognition as sr
import tempfile
import numpy as np
import wave

# --- Streamlit page setup ---
st.set_page_config(page_title="🎙️ TalkPlay – Voice-Controlled Adventure Game", layout="wide")
st.title("🎮 TalkPlay – Voice-Controlled Adventure Game")

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
MODEL = "gemini-2.5-flash"  # You can replace this with e.g. "models/gemini-2.5-flash-live-preview"

# --- Basic game state ---
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

# --- Gemini AI Narration ---
def gemini_reply(context):
    """Get immersive narrative output from Gemini."""
    prompt = f"""
    You are TalkPlay AI, the narrator of an interactive adventure game.
    The player just said: "{context}"
    Continue the story immersively and describe what happens next.
    """
    response = genai.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}]
    )
    return response["choices"][0]["message"]["content"]

# --- Audio Processor for speech-to-text ---
class AudioProcessor(AudioProcessorBase):
    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.text = ""

    def recv_audio(self, frame):
        # Convert audio frame from WebRTC to numpy
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

# Display chat history
for msg in st.session_state.history:
    if msg["role"] == "user":
        st.chat_message("user").markdown(msg["content"])
    else:
        st.chat_message("assistant").markdown(msg["content"])

# --- Voice input section ---
st.subheader("🎤 Speak your command below:")

webrtc_ctx = webrtc_streamer(
    key="speech",
    mode=WebRtcMode.SENDRECV,
    audio_processor_factory=AudioProcessor,
    media_stream_constraints={"audio": True, "video": False},
)

# --- Manual text input fallback ---
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

    # Simple local response
    local_response = process_command(command)

    # Gemini immersive response
    ai_response = gemini_reply(f"{command}. Context: {local_response}")
    full_response = f"{local_response}\n\n**AI Narration:** {ai_response}"

    st.chat_message("assistant").markdown(full_response)
    st.session_state.history.append({"role": "assistant", "content": full_response})

# --- Reset game button ---
if st.button("🔄 Restart Game"):
    st.session_state.history = [{"role": "assistant", "content": GAME_DESCRIPTION}]
    GAME_STATE["location"] = "forest"
    GAME_STATE["inventory"] = []
    GAME_STATE["visited"] = {"forest"}
    st.experimental_rerun()