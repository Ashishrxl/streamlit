import streamlit as st
import tempfile
import asyncio
import websockets
import json
import base64
from google import genai
from google.genai import types
from streamlit.components.v1 import html

# ==============================
# Hide Streamlit elements (same as your original code)
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

st.set_page_config(page_title="🎙️ Live Coaching (Gemini Realtime)", layout="wide")

# ==============================
# Google Gemini setup
# ==============================
if "GOOGLE_API_KEY" not in st.secrets:
    st.error("❌ Missing GOOGLE_API_KEY in Streamlit secrets.")
    st.stop()

API_KEY = st.secrets["GOOGLE_API_KEY"]

REALTIME_URL = (
    "wss://generativelanguage.googleapis.com/ws/"
    "google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent"
    f"?key={API_KEY}"
)

MODEL_NAME = "gemini-2.5-flash-native-audio-dialog"

st.header("🎤 Live Coaching (Gemini Realtime)")
st.write("Stream your voice live and get instant spoken feedback — now with session recording!")

# ==============================
# UI
# ==============================
start = st.button("🚀 Start Live Coaching")
stop = st.button("🟥 Stop & Save Session")

status_box = st.empty()
audio_box = st.empty()

# ==============================
# Helper for WebSocket connection
# ==============================
async def connect_to_gemini(audio_bytes):
    """Send recorded audio chunk to Gemini Live API."""
    async with websockets.connect(REALTIME_URL, ping_interval=None) as ws:
        # Setup message
        setup = {
            "setup": {
                "model": MODEL_NAME,
                "generationConfig": {
                    "responseModalities": ["TEXT", "AUDIO"],
                    "speechConfig": {
                        "voiceConfig": {
                            "prebuiltVoiceConfig": {"voiceName": "Ava"}
                        }
                    }
                }
            }
        }
        await ws.send(json.dumps(setup))

        # Send audio chunk
        encoded = base64.b64encode(audio_bytes).decode()
        await ws.send(json.dumps({"realtimeInput": {"mediaChunks": [{"data": encoded}]}}))

        # Close input stream
        await ws.send(json.dumps({"realtimeInput": {"mediaComplete": True}}))

        # Collect AI responses
        result_text = ""
        result_audio = None

        async for msg in ws:
            message = json.loads(msg)
            if "realtimeOutput" in message:
                parts = message["realtimeOutput"].get("output", {}).get("content", [])
                for part in parts:
                    if "text" in part:
                        result_text += part["text"]
                    elif "inlineData" in part and part["inlineData"]["mimeType"].startswith("audio"):
                        result_audio = base64.b64decode(part["inlineData"]["data"])
                break  # stop after first response

        return result_text, result_audio


# ==============================
# Streamlit interaction logic
# ==============================
if start:
    st.info("🎙️ Ready to record! Speak or sing something for feedback.")
    recorded = st.audio_input("🎤 Record your voice (up to 30 seconds)", key="recorder", label_visibility="collapsed")

    if recorded:
        status_box.info("📡 Sending your audio to Gemini Realtime API...")
        try:
            feedback_text, feedback_audio = asyncio.run(connect_to_gemini(recorded.getvalue()))

            st.subheader("💬 AI Feedback")
            st.write(feedback_text or "No feedback received.")

            if feedback_audio:
                tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name
                with open(tmp, "wb") as f:
                    f.write(feedback_audio)
                st.audio(tmp)
                st.success("✅ Audio feedback ready!")

        except Exception as e:
            st.error(f"⚠️ Connection or API error: {e}")
    else:
        st.warning("⚠️ No recording detected. Please record your voice.")

elif stop:
    st.success("✅ Session stopped and saved!")