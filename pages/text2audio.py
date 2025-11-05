import streamlit as st
from google import genai
from google.genai import types
import wave
from io import BytesIO
import time
from streamlit.components.v1 import html
import base64

# Hide Streamlit branding
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

st.set_page_config(
    page_title="🎙️ Text 2 Audio",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
#MainMenu, footer, [data-testid="stStatusWidget"], [data-testid="stToolbar"] {display: none;}
a[href^="https://github.com"], a[href^="https://streamlit.io"] {display: none !important;}
header > div:nth-child(2) { display: none; }
.main { padding: 2rem; }
.stButton > button {
    width: 100%;
    background-color: #4CAF50;
    color: white;
    padding: 0.75rem;
    font-size: 1.1rem;
}
.success-box {
    padding: 1rem;
    background-color: #d4edda;
    border: 1px solid #c3e6cb;
    border-radius: 0.25rem;
    color: #155724;
}
.warning-box {
    padding: 1rem;
    background-color: #fff3cd;
    border: 1px solid #ffeaa7;
    border-radius: 0.25rem;
    color: #856404;
}
.info-box {
    padding: 1rem;
    background-color: #d1ecf1;
    border: 1px solid #bee5eb;
    border-radius: 0.25rem;
    color: #0c5460;
}
</style>
""", unsafe_allow_html=True)

# Initialize session state
for key, val in {
    'audio_generated': False,
    'audio_buffer': None,
    'summary_text': None,
    'original_word_count': 0,
    'final_word_count': 0,
    'was_summarized': False,
    'selected_voice_used': None,
    'text_confirmed': False,
    'input_text': "",
    'typed_text_temp': "",
    'current_typed_text': ""
}.items():
    if key not in st.session_state:
        st.session_state[key] = val

# Helper: save PCM data to WAV buffer
def save_wave_file(pcm_data, channels=1, rate=24000, sample_width=2):
    buffer = BytesIO()
    with wave.open(buffer, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(sample_width)
        wf.setframerate(rate)
        wf.writeframes(pcm_data)
    buffer.seek(0)
    return buffer

# Extract text from files
def extract_text_from_file(uploaded_file):
    file_type = uploaded_file.name.split('.')[-1].lower()
    try:
        if file_type == 'txt':
            return uploaded_file.read().decode('utf-8')

        elif file_type == 'pdf':
            import PyPDF2
            from io import BytesIO
            pdf_reader = PyPDF2.PdfReader(BytesIO(uploaded_file.read()))
            return " ".join(page.extract_text() for page in pdf_reader.pages)

        elif file_type in ['doc', 'docx']:
            import docx
            from io import BytesIO
            doc = docx.Document(BytesIO(uploaded_file.read()))
            return " ".join(p.text for p in doc.paragraphs)

        else:
            st.error(f"Unsupported file format: {file_type}")
            return None
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
        return None

# Summarize long text
def summarize_text(text, api_key, max_words=3500):
    try:
        client = genai.Client(api_key=api_key)
        prompt = f"""Summarize the following text to under {max_words} words, preserving key ideas.

{text}

Summary:"""
        response = client.models.generate_content(
            model="gemini-2.0-flash-exp",
            contents=prompt
        )
        return response.text
    except Exception as e:
        st.error(f"Error during summarization: {str(e)}")
        return None

# ✅ Fixed Generate Audio function
def generate_audio_tts(text, api_key, voice_name='Kore', speaking_style=''):
    try:
        client = genai.Client(api_key=api_key)
        prompt = f"{speaking_style}: {text}" if speaking_style else text

        response = client.models.generate_content(
            model="gemini-2.5-flash-preview-tts",
            contents=prompt,
            config=types.GenerateContentConfig(
                response_modalities=["AUDIO"],
                speech_config=types.SpeechConfig(
                    voice_config=types.VoiceConfig(
                        prebuilt_voice_config=types.PrebuiltVoiceConfig(
                            voice_name=voice_name,
                        )
                    ),
                )
            )
        )

        # ✅ Safe check and Base64 padding fix
        if (
            hasattr(response, "candidates")
            and response.candidates
            and hasattr(response.candidates[0], "content")
            and response.candidates[0].content
            and response.candidates[0].content.parts
        ):
            audio_part = response.candidates[0].content.parts[0]
            if hasattr(audio_part, "inline_data") and audio_part.inline_data.data:
                b64_data = audio_part.inline_data.data
                # ✅ Fix base64 padding
                missing_padding = len(b64_data) % 4
                if missing_padding:
                    b64_data += '=' * (4 - missing_padding)
                audio_data = base64.b64decode(b64_data)
                return audio_data

        st.error("❌ No audio data returned from Gemini TTS. Check your API key or input text.")
        return None

    except Exception as e:
        st.error(f"Error generating audio: {str(e)}")
        return None

# -------------------------- MAIN APP --------------------------

def main():
    st.title("🎙️ Text-to-Audio Converter")
    st.markdown("### Convert text or documents into natural-sounding speech")
    st.markdown("---")

    MAX_WORDS_FOR_TTS = 4000

    with st.expander("⚙️ Settings", expanded=False):
        st.header("Configuration")
        api_key = st.secrets["GOOGLE_API_KEY"]

        st.subheader("🎵 Voice Options")
        voice_options = {
            'Kore': 'Firm and clear',
            'Puck': 'Upbeat and energetic',
            'Zephyr': 'Bright and friendly',
            'Charon': 'Informative and steady',
            'Fenrir': 'Excitable and dynamic',
            'Aoede': 'Breezy and light',
            'Leda': 'Youthful and vibrant',
            'Orus': 'Firm and authoritative',
            'Callirrhoe': 'Easy-going and relaxed',
            'Autonoe': 'Bright and articulate'
        }
        selected_voice = st.selectbox(
            "Select Voice",
            options=list(voice_options.keys()),
            format_func=lambda x: f"{x} - {voice_options[x]}"
        )

        speaking_style = st.text_input(
            "Optional Speaking Style",
            placeholder="e.g., Say cheerfully, Speak calmly"
        )

        st.markdown("---")
        st.info("💡 Supported formats: TXT, PDF, DOCX")
        st.info(f"📊 Word limit: {MAX_WORDS_FOR_TTS}")
        st.info("🤖 Texts above this limit are automatically summarized")

    col1, col2 = st.columns([1, 1])

    # -------------------------- INPUT --------------------------
    with col1:
        st.header("📝 Input Text")
        input_tab1, input_tab2 = st.tabs(["📁 Upload File", "✍️ Type Text"])

        with input_tab1:
            uploaded_file = st.file_uploader("Upload text file", type=['txt', 'pdf', 'docx', 'doc'])
            if uploaded_file:
                with st.spinner("Extracting text..."):
                    text = extract_text_from_file(uploaded_file)
                if text:
                    st.session_state.input_text = text
                    st.session_state.text_confirmed = True
                    word_count = len(text.split())
                    st.text_area("Extracted Text", text, height=300, disabled=True)
                    st.caption(f"📊 Word count: {word_count}")
                    if word_count > MAX_WORDS_FOR_TTS:
                        st.warning(f"Text exceeds {MAX_WORDS_FOR_TTS} words and will be summarized.")

        with input_tab2:
            if st.session_state.text_confirmed and st.session_state.input_text:
                st.text_area("Confirmed Text", st.session_state.input_text[:500] + "...", disabled=True, height=150)
                if st.button("🔄 Edit Text"):
                    st.session_state.text_confirmed = False
                    st.session_state.audio_generated = False
                    st.session_state.typed_text_temp = st.session_state.input_text
                    st.rerun()
            else:
                typed_text = st.text_area("Type or paste your text here", height=300, value=st.session_state.typed_text_temp)
                if st.button("✅ Confirm Text"):
                    if typed_text.strip():
                        st.session_state.input_text = typed_text
                        st.session_state.text_confirmed = True
                        st.session_state.audio_generated = False
                        st.session_state.typed_text_temp = ""
                        st.success("Text confirmed!")
                        st.rerun()
                    else:
                        st.warning("Please enter text before confirming.")

    # -------------------------- AUDIO --------------------------
    with col2:
        st.header("🔊 Generate Audio")

        if not st.secrets.get("GOOGLE_API_KEY"):
            st.error("Missing Google API Key in Streamlit Secrets.")
            return

        if st.session_state.text_confirmed and st.session_state.input_text:
            text = st.session_state.input_text
            word_count = len(text.split())
            needs_summary = word_count > MAX_WORDS_FOR_TTS

            if st.button("🎵 Convert to Audio"):
                use_text = text
                st.session_state.audio_generated = False
                st.session_state.was_summarized = False

                if needs_summary:
                    with st.spinner("Summarizing text..."):
                        summary = summarize_text(text, st.secrets["GOOGLE_API_KEY"], MAX_WORDS_FOR_TTS)
                    if summary:
                        st.session_state.summary_text = summary
                        st.session_state.was_summarized = True
                        use_text = summary
                        st.success("✅ Text summarized successfully!")
                    else:
                        st.error("Summarization failed.")
                        return

                with st.spinner("Generating audio..."):
                    audio_data = generate_audio_tts(use_text, st.secrets["GOOGLE_API_KEY"], selected_voice, speaking_style)

                if audio_data:
                    buffer = save_wave_file(audio_data)
                    st.session_state.audio_buffer = buffer
                    st.session_state.audio_generated = True
                    st.session_state.final_word_count = len(use_text.split())
                    st.session_state.selected_voice_used = selected_voice
                    st.success("✅ Audio generated successfully!")

            if st.session_state.audio_generated and st.session_state.audio_buffer:
                st.audio(st.session_state.audio_buffer, format="audio/wav")
                timestamp = time.strftime("%Y%m%d-%H%M%S")
                st.download_button(
                    "⬇️ Download Audio",
                    st.session_state.audio_buffer,
                    file_name=f"audio_{timestamp}.wav",
                    mime="audio/wav"
                )
                st.info(f"🎵 Voice: {st.session_state.selected_voice_used} | 📝 Words: {st.session_state.final_word_count}")

                if st.session_state.was_summarized:
                    with st.expander("📄 View Summary"):
                        st.text_area("Summarized Text", st.session_state.summary_text, height=200, disabled=True)
                    st.caption(f"Original: {st.session_state.original_word_count} words")

                if st.button("🔄 Start New Conversion"):
                    for key in list(st.session_state.keys()):
                        del st.session_state[key]
                    st.rerun()
        else:
            st.info("👈 Please upload or type text first.")

if __name__ == "__main__":
    main()