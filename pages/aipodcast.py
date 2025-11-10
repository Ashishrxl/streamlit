import streamlit as st
from google import genai
from google.genai import types
import wave
import base64
from streamlit.components.v1 import html

# --- Hide Streamlit UI elements ---
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

hide_streamlit_style = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
[data-testid="stStatusWidget"] {display: none;}
[data-testid="stToolbar"] {display: none;}
a[href^="https://github.com"] {display: none !important;}
a[href^="https://streamlit.io"] {display: none !important;}
header > div:nth-child(2) {
    display: none;
}
</style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# --- API Key selection ---
api_keys = {
    "Key 1": st.secrets["KEY_1"],
    "Key 2": st.secrets["KEY_2"], "Key 3": st.secrets["KEY_3"], "Key 4": st.secrets["KEY_4"], "Key 5": st.secrets["KEY_5"], "Key 6": st.secrets["KEY_6"], "Key 7": st.secrets["KEY_7"], "Key 8": st.secrets["KEY_8"], "Key 9": st.secrets["KEY_9"], "Key 10": st.secrets["KEY_10"], "Key 11": st.secrets["KEY_11"]
}
selected_key_name = st.selectbox("Select Key", list(api_keys.keys()))
api_key = api_keys[selected_key_name]

# --- Initialize GenAI client with API key from secrets ---
client = genai.Client(api_key=api_key)

# --- Language code mapper ---
def map_language_code(language: str) -> str:
    lang = language.lower()
    if lang == "english":
        return "en-US"
    elif lang == "hindi":
        return "hi-IN"
    elif lang == "bhojpuri":
        return "bho-IN"  # may fallback if unsupported
    return "en-US"

# --- Script Generation ---
def generate_script(topic: str) -> str:
    prompt = f"""
    Write a friendly and engaging podcast script about "{topic}".
    Include:
    - A short intro
    - 3 key talking points
    - A closing statement
    Keep it conversational and natural.
    """
    resp = client.models.generate_content(
        model="gemini-2.5-pro",
        contents=prompt
    )
    return resp.text

# --- WAV saving helper ---
def save_wave(filename: str, pcm_data: bytes, channels=1, rate=24000, sample_width=2):
    with wave.open(filename, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(sample_width)
        wf.setframerate(rate)
        wf.writeframes(pcm_data)

# --- Audio (TTS) generation ---
def generate_audio(script_text: str, voice_name: str = "Kore", language: str = "English") -> str:
    if language.lower() == "hindi":
        style_prompt = "Speak this in a warm and expressive Hindi accent."
    elif language.lower() == "bhojpuri":
        style_prompt = "Speak this in a friendly Bhojpuri tone, like a local storyteller."
    else:
        style_prompt = "Speak this in a natural and friendly tone."

    contents = f"{style_prompt}\n\n{script_text}"

    config = types.GenerateContentConfig(
        response_modalities=["AUDIO"],
        speech_config=types.SpeechConfig(
            language_code=map_language_code(language),
            voice_config=types.VoiceConfig(
                prebuilt_voice_config=types.PrebuiltVoiceConfig(
                    voice_name=voice_name
                )
            )
        )
    )

    response = client.models.generate_content(
        model="gemini-2.5-flash-preview-tts",
        contents=contents,
        config=config
    )

    pcm_data = response.candidates[0].content.parts[0].inline_data.data
    if isinstance(pcm_data, str):
        pcm_data = base64.b64decode(pcm_data)

    filename = "podcast.wav"
    save_wave(filename, pcm_data)
    return filename

# --- Streamlit UI ---
st.set_page_config(page_title="VoiceVerse AI", layout="centered")
st.title("🎙️ VoiceVerse AI Podcast Generator")

topic = st.text_input("Enter your podcast topic:")
language = st.selectbox("Choose a language:", ["English", "Hindi", "Bhojpuri"])
gender = st.radio("Select voice gender:", ["Female", "Male"])

female_voices = ["Kore", "Aoede", "Callirhoe"]
male_voices = ["Puck", "Charon", "Fenrir"]
voice = st.selectbox("Choose a voice:", female_voices if gender == "Female" else male_voices)

if st.button("Generate Podcast"):
    with st.spinner("Creating your podcast script..."):
        script = generate_script(topic)
        st.text_area("Generated Script", script, height=300)

    with st.spinner("Converting to audio..."):
        audio_file = generate_audio(script, voice, language)
        st.audio(audio_file, format="audio/wav")
        with open(audio_file, "rb") as f:
            st.download_button(
                label="📥 Download Podcast",
                data=f,
                file_name="voiceverse_podcast.wav",
                mime="audio/wav"
            )