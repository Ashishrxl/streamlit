import streamlit as st
import google.generativeai as genai
from google.generativeai import types
import wave

# 🔐 Configure Gemini API
genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])

# ✍️ Generate podcast script
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
    response = model.generate_content(prompt)
    return response.text

# 🔊 Save audio to WAV
def save_wave(filename, pcm_data, channels=1, rate=24000, sample_width=2):
    with wave.open(filename, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(sample_width)
        wf.setframerate(rate)
        wf.writeframes(pcm_data)

# 🗣️ Generate audio from script
def generate_audio(script_text, voice_name="Kore", language="English"):
    if language.lower() == "hindi":
        style_prompt = "Speak this in a warm and expressive Hindi accent."
    elif language.lower() == "bhojpuri":
        style_prompt = "Speak this in a friendly Bhojpuri tone, like a local storyteller."
    else:
        style_prompt = "Speak this in a natural and friendly tone."

    model = genai.GenerativeModel("gemini-2.5-pro-preview-tts")
    response = model.generate_content(
        contents=f"{style_prompt}\n\n{script_text}",
        generation_config=types.GenerationConfig(response_mime_type="audio/wav"),
        response_modality="AUDIO",
        speech_config=types.SpeechConfig(
            voice_config=types.VoiceConfig(prebuilt_voice=voice_name)
        )
    )
    audio_data = response.candidates[0].content.parts[0].inline_data.data
    filename = "podcast.wav"
    save_wave(filename, audio_data)
    return filename

# 🎙️ Streamlit UI
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