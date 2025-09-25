import streamlit as st 
import google.generativeai as genai

--- CONFIG ---

GEMMA_MODEL = "gemma-3-12b-it" IMAGEN_MODEL = "imagen-3.0-generate-002" TTS_MODEL = "gemini-2.5-flash-preview-tts"

Load API key from Streamlit secrets

genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])

st.set_page_config(page_title="AI Roleplay + Comics", layout="wide") st.title("🎭 AI Roleplay + Comics Generator")

--- USER INPUTS ---

genre = st.text_input("Enter story genre", "Cyberpunk mystery") characters = st.text_area("List characters (comma separated)", "Detective, Hacker, AI sidekick") length = st.selectbox("Story length", ["Short", "Medium", "Long"]) add_audio = st.checkbox("Generate character voices (TTS)")

if st.button("Generate Story & Comics"): with st.spinner("✨ Writing story with Gemma..."): prompt = f"Write a {length} {genre} roleplay story with characters: {characters}. Split into scenes with dialogue." story_model = genai.GenerativeModel(GEMMA_MODEL) story_resp = story_model.generate_content(prompt) story = story_resp.text st.session_state.story = story

st.subheader("📖 Story Script")
st.write(story)

scenes = story.split("Scene")
for i, scene in enumerate(scenes[1:], start=1):
    st.markdown(f"### 🎬 Scene {i}")
    st.write(scene)

    with st.spinner("🎨 Generating comic panel..."):
        img_prompt = f"Comic panel, {genre}, featuring {characters}. Scene description: {scene[:300]}"
        img_model = genai.GenerativeModel(IMAGEN_MODEL)
        img_resp = img_model.generate_content(img_prompt)
        if hasattr(img_resp, "_result") and hasattr(img_resp._result, "images"):
            img_url = img_resp._result.images[0].url
            st.image(img_url, use_column_width=True)

    if add_audio:
        with st.spinner("🔊 Generating character voices..."):
            tts_model = genai.GenerativeModel(TTS_MODEL)
            tts_resp = tts_model.generate_content(scene, request_options={"output_mime_type": "audio/mpeg"})
            if hasattr(tts_resp, "audio") and len(tts_resp.audio) > 0:
                audio_data = tts_resp.audio[0].data
                audio_file = f"scene_{i}.mp3"
                with open(audio_file, "wb") as f:
                    f.write(audio_data)
                st.audio(audio_file)

st.markdown("---") st.caption("Built with Gemma + Imagen + Gemini TTS via Google Generative AI SDK")

