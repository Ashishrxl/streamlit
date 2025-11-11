import streamlit as st
from google import genai
from google.genai import types
import wave
from io import BytesIO
import time
import base64
from streamlit.components.v1 import html

# Hide Streamlit default elements
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

st.set_page_config(page_title="🎙️ Text 2 Audio", layout="wide")

# CSS to hide unwanted elements
hide_streamlit_style = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
[data-testid="stStatusWidget"] {display: none;}
[data-testid="stToolbar"] {display: none;}
a[href^="https://github.com"] {display: none !important;}
a[href^="https://streamlit.io"] {display: none !important;}
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
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)



# Initialize session state
if 'audio_generated' not in st.session_state:
    st.session_state.audio_generated = False
if 'audio_buffer' not in st.session_state:
    st.session_state.audio_buffer = None
if 'summary_text' not in st.session_state:
    st.session_state.summary_text = None
if 'original_word_count' not in st.session_state:
    st.session_state.original_word_count = 0
if 'final_word_count' not in st.session_state:
    st.session_state.final_word_count = 0
if 'was_summarized' not in st.session_state:
    st.session_state.was_summarized = False
if 'selected_voice_used' not in st.session_state:
    st.session_state.selected_voice_used = None
if 'text_confirmed' not in st.session_state:
    st.session_state.text_confirmed = False
if 'input_text' not in st.session_state:
    st.session_state.input_text = ""
if 'typed_text_temp' not in st.session_state:
    st.session_state.typed_text_temp = ""
if 'current_typed_text' not in st.session_state:
    st.session_state.current_typed_text = ""

# Helper: Save PCM as WAV
def save_wave_file(pcm_data, channels=1, rate=24000, sample_width=2):
    buffer = BytesIO()
    with wave.open(buffer, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(sample_width)
        wf.setframerate(rate)
        wf.writeframes(pcm_data)
    buffer.seek(0)
    return buffer

# Extract text from uploaded file
def extract_text_from_file(uploaded_file):
    file_type = uploaded_file.name.split('.')[-1].lower()
    try:
        if file_type == 'txt':
            return uploaded_file.read().decode('utf-8')
        elif file_type == 'pdf':
            import PyPDF2
            pdf_reader = PyPDF2.PdfReader(BytesIO(uploaded_file.read()))
            return " ".join(page.extract_text() for page in pdf_reader.pages)
        elif file_type in ['doc', 'docx']:
            import docx
            doc = docx.Document(BytesIO(uploaded_file.read()))
            return " ".join(p.text for p in doc.paragraphs)
        else:
            st.error(f"Unsupported file format: {file_type}")
            return None
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
        return None

# Summarize text using Gemini
def summarize_text(text, api_key, max_words=3500):
    try:
        client = genai.Client(api_key=api_key)
        prompt = f"""Please provide a comprehensive summary of the following text.
Keep it under {max_words} words.

TEXT:
{text}
SUMMARY:"""
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt
        )
        return response.text
    except Exception as e:
        st.error(f"Error during summarization: {str(e)}")
        return None

# ✅ Fixed generate_audio_tts() – handles bytes and Base64 correctly
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

        # Safe handling and decoding
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

                # Handle both bytes and str
                if isinstance(b64_data, bytes):
                    audio_data = b64_data
                elif isinstance(b64_data, str):
                    missing_padding = len(b64_data) % 4
                    if missing_padding:
                        b64_data += "=" * (4 - missing_padding)
                    audio_data = base64.b64decode(b64_data)
                else:
                    st.error("Unexpected audio data format from API.")
                    return None

                return audio_data

        st.error("❌ No audio data returned from Gemini TTS. Check your API key or text.")
        return None

    except Exception as e:
        st.error(f"Error generating audio: {str(e)}")
        return None

def main():
    st.title("🎙️ Text-to-Audio Converter")
    st.markdown("### Convert your text files to natural-sounding speech")
    st.markdown("---")

    MAX_WORDS_FOR_TTS = 4000

    # Sidebar settings
    with st.expander("⚙️ Settings", expanded=False):
        st.header("Configuration")
        # --- API Key selection ---
        api_keys = { "Key 1": st.secrets["KEY_1"], "Key 2": st.secrets["KEY_2"], "Key 3": st.secrets["KEY_3"], "Key 4": st.secrets["KEY_4"], "Key 5": st.secrets["KEY_5"], "Key 6": st.secrets["KEY_6"], "Key 7": st.secrets["KEY_7"], "Key 8": st.secrets["KEY_8"], "Key 9": st.secrets["KEY_9"], "Key 10": st.secrets["KEY_10"], "Key 11": st.secrets["KEY_11"]}
        selected_key_name = st.selectbox("Select Key", list(api_keys.keys()))
        api_key = api_keys[selected_key_name]


        st.markdown("---")
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
        selected_voice = st.selectbox("Select Voice", options=list(voice_options.keys()),
                                      format_func=lambda x: f"{x} - {voice_options[x]}")

        st.subheader("🎭 Speaking Style")
        speaking_style = st.text_input("Optional: Describe speaking tone",
                                       placeholder="e.g., Calm and confident")

        st.info("💡 Supported: TXT, PDF, DOCX\n🤖 Auto-summarization for long texts")

    col1, col2 = st.columns([1, 1])

    # Left column: input
    with col1:
        st.header("📝 Input Text")
        tab1, tab2 = st.tabs(["📁 Upload File", "✍️ Type Text"])

        with tab1:
            uploaded = st.file_uploader("Upload text file", type=["txt", "pdf", "docx", "doc"])
            if uploaded:
                with st.spinner("Extracting text..."):
                    extracted = extract_text_from_file(uploaded)
                if extracted:
                    st.session_state.input_text = extracted
                    st.session_state.text_confirmed = True
                    st.text_area("Extracted Text", extracted[:2000], height=300, disabled=True)
                    count = len(extracted.split())
                    st.caption(f"📊 Word count: {count}")
                    if count > MAX_WORDS_FOR_TTS:
                        st.warning(f"⚠️ Text exceeds {MAX_WORDS_FOR_TTS} words — will summarize automatically.")

        with tab2:
            if st.session_state.text_confirmed and st.session_state.input_text and not uploaded:
                wc = len(st.session_state.input_text.split())
                st.success(f"✅ Text confirmed ({wc} words)")
                st.text_area("Confirmed Text", st.session_state.input_text[:500], height=150, disabled=True)
                if st.button("🔄 Edit Text"):
                    st.session_state.text_confirmed = False
                    st.session_state.input_text = ""
                    st.rerun()
            else:
                with st.form("text_form"):
                    text_input = st.text_area("Type or paste your text here", height=300)
                    submitted = st.form_submit_button("✅ Proceed")
                if submitted:
                    if text_input.strip():
                        st.session_state.input_text = text_input
                        st.session_state.text_confirmed = True
                        st.success("Text confirmed! Now generate audio ➡️")
                        st.rerun()
                    else:
                        st.warning("Please enter some text.")

    # Right column: output
    with col2:
        st.header("🔊 Generate Audio")
        if api_key and st.session_state.text_confirmed and st.session_state.input_text:
            txt = st.session_state.input_text
            wc = len(txt.split())
            needs_summary = wc > MAX_WORDS_FOR_TTS

            if st.button("🎵 Convert to Audio"):
                use_text = txt
                if needs_summary:
                    with st.spinner("Summarizing text..."):
                        summary = summarize_text(txt, api_key, max_words=MAX_WORDS_FOR_TTS)
                    if summary:
                        st.session_state.summary_text = summary
                        st.session_state.was_summarized = True
                        use_text = summary
                        st.success("Summarization complete.")
                    else:
                        st.error("Summarization failed.")
                        use_text = None

                if use_text:
                    with st.spinner("Generating audio..."):
                        audio_data = generate_audio_tts(use_text, api_key, selected_voice, speaking_style)
                    if audio_data:
                        audio_buf = save_wave_file(audio_data)
                        st.session_state.audio_generated = True
                        st.session_state.audio_buffer = audio_buf
                        st.audio(audio_buf, format="audio/wav")
                        ts = time.strftime("%Y%m%d-%H%M%S")
                        st.download_button("⬇️ Download Audio", data=audio_buf,
                                           file_name=f"audio_{ts}.wav", mime="audio/wav")
        else:
            st.info("👈 Upload or type text and confirm before generating audio.")

if __name__ == "__main__":
    main()