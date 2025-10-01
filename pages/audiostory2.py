import streamlit as st
import io
import wave
import base64
import time
import os
import random
import uuid

from google import genai
from google.genai import types
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont


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


GEMMA_MODEL = "gemma-3-12b-it"
TTS_MODEL = "gemini-2.5-flash-preview-tts"

# Image generation models in fallback order
IMAGE_MODELS = ["gemini-2.0-flash-exp-image-generation", "gemini-2.0-flash-preview-image-generation", "gemini-2.5-flash-image-preview"]

# IMAGE_MODELS = [ "imagen-3.0-generate-002", "imagen-4.0-generate-preview-06-06", "imagen-4.0-ultra-generate-preview-06-06", "imagen-4.0-generate-001", "imagen-4.0-ultra-generate-001", "imagen-4.0-fast-generate-001", "gemini-2.0-flash-exp-image-generation", "gemini-2.0-flash-preview-image-generation", "gemini-2.5-flash-image-preview"]

# --- API Key selection ---
api_keys = {
    "Key 1": st.secrets["GOOGLE_API_KEY_1"],
    "Key 2": st.secrets["GOOGLE_API_KEY_2"]
}
selected_key_name = st.selectbox("Select API Key", list(api_keys.keys()))
api_key = api_keys[selected_key_name]

client = genai.Client(api_key=api_key)

st.set_page_config(page_title="AI Roleplay Story", layout="wide")
st.title("AI Roleplay Story Generator")

# Inputs
genre = st.text_input("Enter story genre", "Cyberpunk mystery")
characters = st.text_area("List characters (comma separated)", "Detective, Hacker, AI sidekick")
length = st.selectbox("Story length", ["Short", "Medium", "Long"])
language = st.selectbox("Select story language", ["English", "Hindi", "Bhojpuri"])

# Voice choices per language
voice_options = {
    "English": ["English Male", "English Female"],
    "Hindi": ["Hindi Male", "Hindi Female"],
    "Bhojpuri": ["Bhojpuri Male", "Bhojpuri Female"]
}
voice_choice = st.selectbox("Select voice", voice_options[language])

# User options
add_audio = st.checkbox("Generate audio of full story")
add_images = st.checkbox("Generate images for each scene")

# Stop image generation flag
if "stop_images" not in st.session_state:
    st.session_state["stop_images"] = False

# Placeholder for dynamic stop button
stop_button_placeholder = st.empty()

def pcm_to_wav_bytes(pcm_bytes, channels=1, rate=24000, sample_width=2):
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(sample_width)
        wf.setframerate(rate)
        wf.writeframes(pcm_bytes)
    buf.seek(0)
    return buf.read()

def map_voice(voice_choice):
    mapping = {
        "English Male": "Kore",
        "English Female": "Charon",
        "Hindi Male": "Kore",
        "Hindi Female": "Charon",
        "Bhojpuri Male": "Kore",
        "Bhojpuri Female": "Charon"
    }
    return mapping.get(voice_choice, "Kore")

def map_language_code(language):
    codes = {
        "English": "en-US",
        "Hindi": "hi-IN",
        "Bhojpuri": "bh-IN"
    }
    return codes.get(language, "en-US")

def generate_pdf_unicode(text, title="AI Roleplay Story"):
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    width, height = A4

    font_path = "NotoSansDevanagari-Regular.ttf"
    if not os.path.exists(font_path):
        raise FileNotFoundError("Add NotoSansDevanagari-Regular.ttf in the folder for Hindi/Unicode support.")
    pdfmetrics.registerFont(TTFont("NotoSans", font_path))

    y = height - 50
    c.setFont("NotoSans", 18)
    c.drawString(50, y, title)
    y -= 30

    c.setFont("NotoSans", 12)
    for line in text.split("\n"):
        if y < 50:
            c.showPage()
            c.setFont("NotoSans", 12)
            y = height - 50
        c.drawString(50, y, line)
        y -= 18

    c.showPage()
    c.save()
    buf.seek(0)
    return buf

def safe_generate_image(prompt, retries=2, delay=5):
    for model in IMAGE_MODELS:
        for attempt in range(retries):
            if st.session_state.get("stop_images", False):
                return None
            try:
                img_resp = client.models.generate_content(
                    model=model,
                    contents=[prompt]
                )
                return img_resp
            except Exception as e:
                stop_button_placeholder.button(
                    "Stop Image Generation",
                    key=f"stop_button_{uuid.uuid4()}",
                    on_click=lambda: st.session_state.update({"stop_images": True})
                )
                if attempt < retries - 1:
                    wait = delay + random.randint(0,3)
                    st.warning(f"{model} overloaded. Retrying in {wait}s...")
                    time.sleep(wait)
                else:
                    st.warning(f"{model} failed. Trying next model...")
                    break
    return None

def generate_images_from_story(story_text):
    scenes = [p.strip() for p in story_text.split("\n") if p.strip()]
    images = []
    st.session_state["stop_images"] = False
    stop_button_placeholder.empty()

    for i, scene in enumerate(scenes, 1):
        if st.session_state["stop_images"]:
            st.warning("Image generation stopped by user.")
            break

        img_resp = safe_generate_image(
            f"Create a high-quality illustration for this scene:\n{scene}"
        )
        if img_resp and hasattr(img_resp, "candidates") and img_resp.candidates:
            candidate = img_resp.candidates[0]
            if hasattr(candidate, "content") and candidate.content:
                for part in candidate.content.parts:
                    if hasattr(part, "inline_data") and hasattr(part.inline_data, "data"):
                        img_bytes = base64.b64decode(part.inline_data.data)
                        images.append((i, img_bytes))
                        break
    return images

# --- Main: Generate story + audio + images ---
if st.button("Generate Story & Audio"):

    # --- Story generation ---
    story_progress = st.progress(0)
    story_placeholder = st.empty()
    prompt = (
        f"Write a {length} {genre} roleplay story in {language} ONLY. "
        f"Include first a brief introduction of each character ({characters}) and then the story. "
        f"Do NOT include text in any other language. "
        f"Do NOT include explanations, summaries, or extra content. "
        f"The output must be entirely in {language} and contain ONLY character introductions and the story."
    )

    for pct in range(0, 101, 5):
        story_progress.progress(pct, text=f"Generating story ~{100-pct}s left")
        story_placeholder.write(f"⏳ Generating story (about {100-pct}s remaining)")
        time.sleep(0.1)

    resp = client.models.generate_content(model=GEMMA_MODEL, contents=[prompt])
    story = getattr(resp, "text", str(resp))
    st.session_state["story"] = story

    story_progress.progress(100, text="Story generated ✅")
    story_placeholder.write("✅ Story ready!")

    # --- Audio generation ---
    if add_audio:
        audio_progress = st.progress(0)
        audio_placeholder = st.empty()
        for pct in range(0, 101, 5):
            audio_progress.progress(pct, text=f"Generating audio ~{100-pct}s left")
            audio_placeholder.write(f"⏳ Generating audio (about {100-pct}s remaining)")
            time.sleep(0.1)

        config = types.GenerateContentConfig(
            response_modalities=["AUDIO"],
            speech_config=types.SpeechConfig(
                language_code=map_language_code(language),
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(
                        voice_name=map_voice(voice_choice)
                    )
                )
            )
        )

        tts_resp = client.models.generate_content(
            model=TTS_MODEL,
            contents=[st.session_state["story"]],
            config=config
        )

        data = None
        if hasattr(tts_resp, "candidates") and tts_resp.candidates:
            candidate = tts_resp.candidates[0]
            if hasattr(candidate, "content") and candidate.content:
                for part in candidate.content.parts:
                    if hasattr(part, "inline_data") and hasattr(part.inline_data, "data"):
                        data = part.inline_data.data
                        break

        audio_progress.progress(100, text="Audio generated ✅")
        audio_placeholder.write("✅ Audio ready!")

        if data:
            if isinstance(data, str):
                pcm = base64.b64decode(data)
            else:
                pcm = bytes(data)
            wav_bytes = pcm_to_wav_bytes(pcm)
            st.session_state["audio_bytes"] = wav_bytes

    # --- Image generation ---
    if add_images:
        with st.spinner("Generating illustrations for each scene..."):
            images = generate_images_from_story(st.session_state["story"])
            if images:
                st.subheader("Story Illustrations")
                for i, img_bytes in images:
                    st.image(img_bytes, caption=f"Illustration for Scene {i}")

# --- Display story persistently ---
if "story" in st.session_state:
    st.subheader("Story Script")
    st.write(st.session_state["story"])
    try:
        pdf_buffer = generate_pdf_unicode(st.session_state["story"])
        st.download_button(
            label="Download Story as PDF",
            data=pdf_buffer,
            file_name="story.pdf",
            mime="application/pdf"
        )
    except Exception as e:
        st.error(f"PDF generation failed: {e}")

# --- Display audio persistently ---
if "audio_bytes" in st.session_state:
    st.audio(st.session_state["audio_bytes"], format="audio/wav")
    st.download_button(
        label="Download Audio",
        data=st.session_state["audio_bytes"],
        file_name="story_audio.wav",
        mime="audio/wav"
    )

st.markdown("---")
