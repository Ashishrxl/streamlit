import streamlit as st
import asyncio
import numpy as np
from google import genai
from google.genai.types import LiveConnectConfig, Part, Content
from streamlit_webrtc import webrtc_streamer, WebRtcMode
import av

st.set_page_config(page_title="Voice Translator: Hindi ↔ English", page_icon="🗣️", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
.main-header{text-align:center;color:#4A90E2;padding:20px;border-bottom:3px solid #E67E22;}
.user-section{padding:20px;border-radius:10px;background-color:#f8f9fa;margin:10px;}
.status-indicator{padding:10px;border-radius:5px;text-align:center;font-weight:bold;margin:10px 0;}
.status-ready{background-color:#28a745;color:white;}
.status-listening{background-color:#ffc107;color:black;}
.status-translating{background-color:#17a2b8;color:white;}
.status-speaking{background-color:#e67e22;color:white;}
</style>
""", unsafe_allow_html=True)

if 'connected' not in st.session_state:
    st.session_state.connected = False
if 'translations' not in st.session_state:
    st.session_state.translations = []
if 'user1_status' not in st.session_state:
    st.session_state.user1_status = "Ready"
if 'user2_status' not in st.session_state:
    st.session_state.user2_status = "Ready"

with st.sidebar:
    st.header("⚙️ Configuration")
    api_key = st.text_input("Google API Key", type="password")
    model_choice = st.selectbox("Select Model", ["gemini-2.5-flash-native-audio-latest (Recommended)", "gemini-live-2.5-flash-preview", "gemini-2.5-flash-native-audio-preview-09-2025"])
    st.divider()
    st.subheader("📊 Audio Settings")
    st.text("Input: 16-bit PCM, 16kHz")
    st.text("Output: 16-bit PCM, 24kHz")
    if st.button("🔌 Connect to Gemini API", use_container_width=True):
        if api_key:
            st.session_state.connected = True
            st.success("✅ Connected!")
        else:
            st.error("⚠️ Please enter API key")
    if st.session_state.connected:
        st.success("🟢 Connected")
    else:
        st.warning("🔴 Disconnected")
    st.divider()
    st.subheader("📊 Session Stats")
    st.metric("Total Translations", len(st.session_state.translations))
    st.metric("Average Latency", "~250ms")
    st.divider()
    st.subheader("ℹ️ How It Works")
    st.info("1. Speak into your microphone 2. Gemini Live API transcribes in real-time 3. Text is translated 4. Translated audio plays automatically")

st.markdown('<h1 class="main-header">🗣️ Real-Time Voice Translator: Hindi ↔ English</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align:center;color:#666;">Powered by Google Gemini Live API</p>', unsafe_allow_html=True)

col1, col_center, col2 = st.columns([5, 1, 5])

with col1:
    st.markdown('<div class="user-section">', unsafe_allow_html=True)
    st.markdown("### 👤 User 1: Hindi Speaker")
    status_class = f"status-{st.session_state.user1_status.lower()}"
    st.markdown(f'<div class="status-indicator {status_class}">{st.session_state.user1_status}</div>', unsafe_allow_html=True)
    lang1 = st.selectbox("Language", ["Hindi", "English"], key="lang1")
    if st.button("🎤 Press to Speak", key="user1_btn", use_container_width=True):
        st.session_state.user1_status = "Listening"
        st.rerun()
    st.subheader("📝 Transcription")
    st.text_area("Your speech", "", height=100, key="trans1", disabled=True)
    st.subheader("🔄 Translation")
    st.text_area("Translated text", "", height=100, key="transl1", disabled=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col_center:
    st.markdown("<br><br><br>", unsafe_allow_html=True)
    st.markdown('<h1 style="text-align:center;color:#4A90E2;">⇄</h1>', unsafe_allow_html=True)
    if st.session_state.connected:
        st.markdown('<p style="text-align:center;color:#28a745;">●</p>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="user-section">', unsafe_allow_html=True)
    st.markdown("### 👤 User 2: English Speaker")
    status_class = f"status-{st.session_state.user2_status.lower()}"
    st.markdown(f'<div class="status-indicator {status_class}">{st.session_state.user2_status}</div>', unsafe_allow_html=True)
    lang2 = st.selectbox("Language", ["English", "Hindi"], key="lang2")
    if st.button("🎤 Press to Speak", key="user2_btn", use_container_width=True):
        st.session_state.user2_status = "Listening"
        st.rerun()
    st.subheader("📝 Transcription")
    st.text_area("Your speech", "", height=100, key="trans2", disabled=True)
    st.subheader("🔄 Translation")
    st.text_area("Translated text", "", height=100, key="transl2", disabled=True)
    st.markdown('</div>', unsafe_allow_html=True)

st.divider()
st.subheader("📋 Activity Log")
if st.session_state.translations:
    activity_text = "
".join([f"• {t}" for t in st.session_state.translations[-10:]])
    st.text_area("Recent translations", activity_text, height=150, disabled=True)

st.divider()
st.markdown("""
<div style="text-align:center;color:#666;padding:20px;">
<p>Built with Streamlit | Powered by Google Gemini Live API</p>
<p>⚠️ Demo Simulation Only</p>
</div>
""", unsafe_allow_html=True)