import streamlit as st
from google import genai
from google.genai import types
import wave
from io import BytesIO
import time
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

st.set_page_config(
    page_title="🎙️ Text 2 Audio",
    layout="wide"
)

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

# Initialize session state variables
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

# Helper function: save PCM data as WAV in an in-memory buffer
def save_wave_file(pcm_data, channels=1, rate=24000, sample_width=2):
    buffer = BytesIO()
    with wave.open(buffer, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(sample_width)
        wf.setframerate(rate)
        wf.writeframes(pcm_data)
    buffer.seek(0)
    return buffer

# Extract text depending on uploaded file type
def extract_text_from_file(uploaded_file):
    file_type = uploaded_file.name.split('.')[-1].lower()

    try:
        if file_type == 'txt':
            text = uploaded_file.read().decode('utf-8')
            return text

        elif file_type == 'pdf':
            import PyPDF2
            from io import BytesIO

            pdf_reader = PyPDF2.PdfReader(BytesIO(uploaded_file.read()))
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + " "
            return text

        elif file_type in ['doc', 'docx']:
            import docx
            from io import BytesIO

            doc = docx.Document(BytesIO(uploaded_file.read()))
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + " "
            return text

        else:
            st.error(f"Unsupported file format: {file_type}")
            return None

    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
        return None

# Summarize text using Gemini API
def summarize_text(text, api_key, max_words=3500):
    """
    Summarize long text to fit within TTS token limits
    """
    try:
        client = genai.Client(api_key=api_key)
        
        prompt = f"""Please provide a comprehensive summary of the following text. 
Capture all key points, main ideas, and important details while keeping the summary under {max_words} words.
Maintain the flow and context of the original content.

TEXT TO SUMMARIZE:
{text}

SUMMARY:"""

        response = client.models.generate_content(
            model="gemini-2.0-flash-exp",  # or "gemini-1.5-pro" or "gemini-1.5-flash"
            contents=prompt
        )
        
        summary = response.text
        return summary

    except Exception as e:
        st.error(f"Error during summarization: {str(e)}")
        return None

# Generate audio from text using Gemini 2.5 Flash TTS
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

        audio_data = response.candidates[0].content.parts[0].inline_data.data
        return audio_data

    except Exception as e:
        st.error(f"Error generating audio: {str(e)}")
        return None

def main():
    st.title("🎙️ Text-to-Audio Converter")
    st.markdown("### Convert your text files to natural-sound")
    st.markdown("---")

    # Configuration constants
    MAX_WORDS_FOR_TTS = 4000  # Safe limit for TTS without hitting quota

    # Sidebar for configuration
    with st.expander("⚙️ Settings", expanded=False):
        st.header("Configuration")

        api_key = st.secrets["GOOGLE_API_KEY"]

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

        selected_voice = st.selectbox(
            "Select Voice",
            options=list(voice_options.keys()),
            format_func=lambda x: f"{x} - {voice_options[x]}"
        )

        st.subheader("🎭 Speaking Style")
        speaking_style = st.text_input(
            "Optional: Describe how to speak",
            placeholder="e.g., Say cheerfully, Speak in a calm voice",
            help="Leave empty for natural speech"
        )

        st.markdown("---")
        st.info("💡 **Supported file formats:** TXT, PDF, DOCX")
        st.info(f"📊 **Word limit for direct conversion:** {MAX_WORDS_FOR_TTS} words")
        st.info("🤖 **Auto-summarization:** Texts longer than the limit will be automatically summarized")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.header("📝 Input Text")

        input_tab1, input_tab2 = st.tabs(["📁 Upload File", "✍️ Type Text"])

        with input_tab1:
            uploaded_file = st.file_uploader(
                "Upload your text file",
                type=['txt', 'pdf', 'docx', 'doc'],
                help="Upload a text file to convert to audio"
            )

            if uploaded_file is not None:
                st.success(f"✅ File uploaded: {uploaded_file.name}")

                with st.spinner("Extracting text from file..."):
                    extracted_text = extract_text_from_file(uploaded_file)

                if extracted_text:
                    st.session_state.input_text = extracted_text
                    st.session_state.text_confirmed = True  # Auto-confirm for file uploads
                    
                    st.text_area(
                        "Extracted Text",
                        value=extracted_text,
                        height=300,
                        key="extracted_display",
                        disabled=True
                    )

                    word_count = len(extracted_text.split())
                    st.caption(f"📊 Word count: {word_count} words")
                    
                    if word_count > MAX_WORDS_FOR_TTS:
                        st.markdown(
                            f'<div class="warning-box">⚠️ Text exceeds {MAX_WORDS_FOR_TTS} words. '
                            f'It will be automatically summarized before audio conversion.</div>',
                            unsafe_allow_html=True
                        )

        with input_tab2:
            # Display confirmed text or text area
            if st.session_state.text_confirmed and st.session_state.input_text and not uploaded_file:
                confirmed_word_count = len(st.session_state.input_text.split())
                st.markdown(
                    f'<div class="info-box">✅ Text confirmed ({confirmed_word_count} words). '
                    f'Ready to generate audio! ➡️</div>',
                    unsafe_allow_html=True
                )
                
                # Show word count warning if needed
                if confirmed_word_count > MAX_WORDS_FOR_TTS:
                    st.markdown(
                        f'<div class="warning-box">⚠️ Text exceeds {MAX_WORDS_FOR_TTS} words. '
                        f'It will be automatically summarized before audio conversion.</div>',
                        unsafe_allow_html=True
                    )
                
                # Show preview of confirmed text
                st.text_area(
                    "Confirmed Text (preview)",
                    value=st.session_state.input_text[:500] + ("..." if len(st.session_state.input_text) > 500 else ""),
                    height=150,
                    disabled=True,
                    key="confirmed_text_preview"
                )
                
                if st.button("🔄 Edit/Change Text", type="secondary", key="edit_text_btn"):
                    st.session_state.text_confirmed = False
                    st.session_state.typed_text_temp = st.session_state.input_text
                    st.session_state.audio_generated = False
                    st.rerun()
            else:
                # Use form for better mobile experience - immediate button response
                with st.form(key="text_input_form", clear_on_submit=False):
                    typed_text = st.text_area(
                        "Type or paste your text here",
                        height=300,
                        placeholder="Enter the text you want to convert to audio...",
                        value=st.session_state.typed_text_temp,
                        key="typed_input_form_area"
                    )

                    # Form buttons - these respond immediately
                    col_btn1, col_btn2 = st.columns([1, 1])
                    
                    with col_btn1:
                        submit_button = st.form_submit_button("✅ Proceed with This Text", type="primary")
                    
                    with col_btn2:
                        clear_button = st.form_submit_button("🔄 Clear Text", type="secondary")
                
                # Handle form submission
                if submit_button:
                    if typed_text and len(typed_text.strip()) > 0:
                        st.session_state.input_text = typed_text
                        st.session_state.text_confirmed = True
                        st.session_state.audio_generated = False  # Reset audio if new text
                        st.session_state.typed_text_temp = ""
                        st.session_state.current_typed_text = ""
                        st.success("✅ Text confirmed! You can now generate audio.")
                        st.rerun()
                    else:
                        st.warning("⚠️ Please enter some text before proceeding.")
                
                if clear_button:
                    st.session_state.input_text = ""
                    st.session_state.text_confirmed = False
                    st.session_state.audio_generated = False
                    st.session_state.typed_text_temp = ""
                    st.session_state.current_typed_text = ""
                    st.rerun()

    with col2:
        st.header("🔊 Generate Audio")

        # Check if text is confirmed before allowing audio generation
        if api_key and st.session_state.text_confirmed and st.session_state.input_text:
            input_text = st.session_state.input_text
            word_count = len(input_text.split())

            # Determine if summarization is needed
            needs_summarization = word_count > MAX_WORDS_FOR_TTS

            if needs_summarization and not st.session_state.audio_generated:
                st.info(f"📝 Original text: {word_count} words " f"🤖 Will be summarized to ~{MAX_WORDS_FOR_TTS} words before conversion")

            if st.button("🎵 Convert to Audio", type="primary", key="convert_audio_btn"):
                use_text = input_text
                
                # Reset audio generated state
                st.session_state.audio_generated = False
                st.session_state.was_summarized = False
                
                # Summarize if needed
                if needs_summarization:
                    st.markdown("### 🤖 Step 1: Summarizing Text")
                    with st.spinner("Summarizing long text to fit TTS limits... This may take a moment."):
                        summary = summarize_text(input_text, api_key=api_key, max_words=MAX_WORDS_FOR_TTS)
                    
                    if summary:
                        summary_word_count = len(summary.split())
                        st.success(f"✅ Summarization complete! Reduced from {word_count} to {summary_word_count} words")
                        
                        # Store summary in session state
                        st.session_state.summary_text = summary
                        st.session_state.original_word_count = word_count
                        st.session_state.was_summarized = True
                        
                        use_text = summary
                    else:
                        st.error("❌ Failed to summarize text. Please try with shorter text")
                        use_text = None

                # Generate audio if we have valid text
                if use_text:
                    st.markdown("### 🎙️ Step 2: Generating Audio" if needs_summarization else "### 🎙️ Generating Audio")
                    with st.spinner("Converting text to audio... This may take a moment."):
                        audio_data = generate_audio_tts(
                            text=use_text,
                            api_key=api_key,
                            voice_name=selected_voice,
                            speaking_style=speaking_style
                        )

                    if audio_data:
                        audio_buffer = save_wave_file(audio_data)
                        
                        # Store in session state
                        st.session_state.audio_buffer = audio_buffer
                        st.session_state.audio_generated = True
                        st.session_state.final_word_count = len(use_text.split())
                        st.session_state.selected_voice_used = selected_voice

            # Display generated audio (persists after download button click)
            if st.session_state.audio_generated and st.session_state.audio_buffer:
                st.markdown('<div class="success-box">✅ Audio generated successfully!</div>', unsafe_allow_html=True)
                
                # Show summary if it was created
                if st.session_state.was_summarized and st.session_state.summary_text:
                    with st.expander("📄 View Summarized Text", expanded=False):
                        st.text_area(
                            "Summary",
                            value=st.session_state.summary_text,
                            height=200,
                            disabled=True,
                            key="summary_display_persist"
                        )
                
                # Reset buffer position to beginning for audio player
                st.session_state.audio_buffer.seek(0)
                st.audio(st.session_state.audio_buffer, format='audio/wav')

                timestamp = time.strftime("%Y%m%d-%H%M%S")
                filename = f"audio_output_{timestamp}.wav"

                # Reset buffer position again for download
                st.session_state.audio_buffer.seek(0)
                st.download_button(
                    label="⬇️ Download Audio File",
                    data=st.session_state.audio_buffer,
                    file_name=filename,
                    mime="audio/wav",
                    key="download_audio_btn"
                )

                st.info(f"🎵 Voice: {st.session_state.selected_voice_used} | 📝 Words converted: {st.session_state.final_word_count}")
                
                if st.session_state.was_summarized:
                    st.caption(f"ℹ️ Original text ({st.session_state.original_word_count} words) was summarized for audio conversion")
                
                # Add a button to clear and generate new audio
                if st.button("🔄 Generate New Audio", type="secondary", key="new_audio_btn"):
                    st.session_state.audio_generated = False
                    st.session_state.audio_buffer = None
                    st.session_state.summary_text = None
                    st.session_state.was_summarized = False
                    st.session_state.text_confirmed = False
                    st.session_state.input_text = ""
                    st.session_state.typed_text_temp = ""
                    st.session_state.current_typed_text = ""
                    st.rerun()

        else:
            st.info("👈 Please provide:")
            if not api_key:
                st.warning("🔑 API key not found")
            if not st.session_state.text_confirmed or not st.session_state.input_text:
                st.warning("📝 Upload a file or type text and click '✅ Proceed with This Text'")

if __name__ == "__main__":
    main()