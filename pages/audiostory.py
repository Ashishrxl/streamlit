import streamlit as st
from google import genai
from google.genai import types
import io
import wave
import base64

GEMMA_MODEL = "gemma-3-12b-it"
IMAGEN_MODEL = "imagen-3.0-generate-002"
TTS_MODEL = "gemini-2.5-flash-preview-tts"

client = genai.Client(api_key=st.secrets["GOOGLE_API_KEY"])

st.set_page_config(page_title="AI Roleplay + Comics", layout="wide")
st.title("AI Roleplay + Comics Generator")

genre = st.text_input("Enter story genre", "Cyberpunk mystery")
characters = st.text_area("List characters (comma separated)", "Detective, Hacker, AI sidekick")
length = st.selectbox("Story length", ["Short", "Medium", "Long"])
add_audio = st.checkbox("Generate character voices (TTS)")

def pcm_to_wav_bytes(pcm_bytes, channels=1, rate=24000, sample_width=2):
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(sample_width)
        wf.setframerate(rate)
        wf.writeframes(pcm_bytes)
    buf.seek(0)
    return buf.read()

if st.button("Generate Story & Comics"):
    with st.spinner("Writing story..."):
        prompt = f"Write a {length} {genre} roleplay story with characters: {characters}. Split into scenes with dialogue."
        resp = client.models.generate_content(model=GEMMA_MODEL, contents=prompt)
        story = resp.text if hasattr(resp, "text") else str(resp)
        st.session_state["story"] = story

    st.subheader("Story Script")
    st.write(story)

    scenes = story.split("Scene")
    for i, scene in enumerate(scenes[1:], start=1):
        st.markdown(f"### Scene {i}")
        st.write(scene)

        with st.spinner("Generating image..."):
            img_resp = client.models.generate_images(model=IMAGEN_MODEL, prompt=scene, config=types.GenerateImagesConfig(number_of_images=1))
            if hasattr(img_resp, "generated_images") and len(img_resp.generated_images) > 0:
                gen_img = img_resp.generated_images[0]
                img_obj = getattr(gen_img, "image", None)
                try:
                    st.image(img_obj, use_column_width=True)
                except Exception:
                    try:
                        img_bytes = getattr(gen_img.image, "imageBytes", None) or getattr(gen_img.image, "image_bytes", None)
                        if img_bytes:
                            st.image(io.BytesIO(img_bytes), use_column_width=True)
                    except Exception:
                        pass

        if add_audio:
            with st.spinner("Generating audio..."):
                config = types.GenerateContentConfig(
                    response_modalities=["AUDIO"],
                    speech_config=types.SpeechConfig(
                        voice_config=types.VoiceConfig(
                            prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name="Kore")
                        )
                    )
                )
                tts_resp = client.models.generate_content(model=TTS_MODEL, contents=scene, config=config)
                try:
                    data = tts_resp.candidates[0].content.parts[0].inline_data.data
                except Exception:
                    data = None
                if data:
                    if isinstance(data, str):
                        pcm = base64.b64decode(data)
                    else:
                        pcm = bytes(data)
                    wav_bytes = pcm_to_wav_bytes(pcm)
                    st.audio(wav_bytes, format="audio/wav")

st.markdown("---")
st.caption("Built with Gemma + Imagen + Gemini TTS")