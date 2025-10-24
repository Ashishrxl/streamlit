import os
import streamlit as st
from streamlit_webrtc import webrtc_streamer, AudioProcessorBase, WebRtcMode
import google.generativeai as genai
import speech_recognition as sr
import tempfile

# --- CONFIG ---
st.set_page_config(page_title="🎙️ TalkPlay – Voice-Controlled Game", layout="wide")

# Use Streamlit secrets for API key
API_KEY = st.secrets["GOOGLE_API_KEY"]
genai.configure(api_key=API_KEY)
MODEL = "gemini-2.5-flash"  # can change to your preferred model

# --- Game logic (simple text adventure) ---
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
    else:
        return f"The forest seems quiet... Your command '{cmd}' doesn't do much."

# --- Gemini integration ---
def gemini_reply(context):
    prompt = f"""
    You are TalkPlay AI, a narrator for an interactive voice-controlled adventure game.
    The player just issued this command: "{context}"
    Respond narratively to describe what happens next. Keep it immersive.
    """
    response = genai.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}]
    )
    return response["choices"][0]["message"]["content"]

# --- Audio Processor ---
class AudioProcessor(AudioProcessorBase):
    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.text = ""

    def recv_audio(self, frame):
        # Convert audio to text using SpeechRecognition
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav") as temp_wav:
                temp_wav.write(frame.to_ndarray().tobytes())
                temp_wav.seek(0)
                with sr.AudioFile(temp_wav.name) as source:
                    audio_data = self.recognizer.record(source)
                    text = self.recognizer.recognize_google(audio_data)
                    self.text = text
        except Exception:
            pass
        return frame

# --- Streamlit UI ---
st.title("🎮 TalkPlay – Voice Controlled Adventure Game")

st.markdown("""
Speak commands like:
- “Go north”
- “Look around”
- “Check inventory”
- “Attack monster”
""")

if "history" not in st.session_state:
    st.session_state.history = [{"role": "assistant", "content": GAME_DESCRIPTION}]

# Display history
for msg in st.session_state.history:
    if msg["role"] == "user":
        st.chat_message("user").markdown(msg["content"])
    else:
        st.chat_message("assistant").markdown(msg["content"])

# --- Voice input via streamlit-webrtc ---
st.subheader("🎙️ Speak your command below:")

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

    # Simple local game reaction
    local_response = process_command(command)

    # Get immersive AI narration
    ai_response = gemini_reply(f"{command}. Context: {local_response}")
    full_response = f"{local_response}\n\n**AI Narration:** {ai_response}"

    st.chat_message("assistant").markdown(full_response)
    st.session_state.history.append({"role": "assistant", "content": full_response})

# Reset button
if st.button("🔄 Restart Game"):
    st.session_state.history = [{"role": "assistant", "content": GAME_DESCRIPTION}]
    GAME_STATE["location"] = "forest"
    GAME_STATE["inventory"] = []
    GAME_STATE["visited"] = {"forest"}
    st.experimental_rerun()