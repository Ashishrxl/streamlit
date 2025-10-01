import streamlit as st
import google.generativeai as genai
# from google.generativeai import types
from google.genai import types
import wave
import base64
from streamlit.components.v1 import html

# --- Hide Streamlit UI elements (original code kept intact) ---
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

/* The following specifically targets and hides all child elements of the header's right side,
   while preserving the header itself and, by extension, the sidebar toggle button. */
header > div:nth-child(2) {
    display: none;
}
</style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)
# -------------------------------------------------------------------

# 🔐 Configure Gemini API
genai.configure(api_key=st.secrets["GOOGLE_API_KEY_2"])

# --- Language code mapper ---
def map_language_code(language: str) -> str:
    lang = language.lower()
    if lang == "english":
        return "en-US"
    elif lang == "hindi":
        return "hi-IN"
    elif lang == "bhojpuri":
        return "bho-IN"  # Bhojpuri is not always officially supported; fallback may apply
    return "en-US"  # default

# --- Script Generation ---
def generate_script(topic):
    prompt = f"""
    Write a friendly and engaging podcast script about "{topic}".
    Include:
    - A short intro
    - 3 key talking points
    - A closing statement
    Keep it conversational and natural.
    """
    model = genai.GenerativeModel("gemini-2.5-pro")
    resp = model.generate_content(prompt)
    return resp.text

# --- Save WAV helper ---
def save_wave(filename, pcm_data, channels=1, rate=24000, sample_width=2):
    with wave.open(filename, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(sample_width)
        wf.setframerate(rate)
        wf.writeframes(pcm_data)

# --- Audio (TTS) Generation ---
def generate_audio(script_text, voice_name="Kore", language="English"):
    # choose style prompt based on language
    if language.lower() == "hindi":
        style_prompt = "Speak this in a warm and expressive Hindi accent."
    elif language.lower() == "bhojpuri":
        style_prompt = "Speak this in a friendly Bhojpuri tone, like a local storyteller."
    else:
        style_prompt = "Speak this in a natural and friendly tone."

    model = genai.GenerativeModel("gemini-2.5-flash-preview-tts")
    contents = f"{style_prompt}\n\n{script_text}"

    # ✅ TTS config style updated with mapped language codes
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

    response = model.generate_content(
        contents=contents,
        config=config
    )

    pcm_data = response.candidates[0].content.parts[0].inline_data.data

    # decode if base64 string
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