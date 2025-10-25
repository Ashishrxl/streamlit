import streamlit as st
from google import genai
from google.genai import types
import wave
from io import BytesIO
import time

# Streamlit page config
st.set_page_config(
    page_title="Text-to-Audio Converter",
    page_icon="🎙️",
    layout="wide",
)

# Custom CSS for styling
st.markdown(
    """
    <style>
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
    </style>
    """,
    unsafe_allow_html=True,
)

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
    st.markdown("### Convert your text files to natural-sounding audio using Google's Gemini AI")
    st.markdown("---")

    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration")

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
        st.info("📊 **Free tier limits:** 3 RPM, 15 RPD for TTS model")

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
                    st.session_state['input_text'] = extracted_text
                    st.text_area(
                        "Extracted Text (editable)",
                        value=extracted_text[:2000] + ("..." if len(extracted_text) > 2000 else ""),
                        height=300,
                        key="extracted_display"
                    )

                    word_count = len(extracted_text.split())
                    st.caption(f"📊 Word count: {word_count} words")

        with input_tab2:
            typed_text = st.text_area(
                "Type or paste your text here",
                height=300,
                placeholder="Enter the text you want to convert to audio...",
                key="typed_input"
            )

            if typed_text:
                st.session_state['input_text'] = typed_text
                word_count = len(typed_text.split())
                st.caption(f"📊 Word count: {word_count} words")

    with col2:
        st.header("🔊 Generate Audio")

        if api_key and 'input_text' in st.session_state and st.session_state['input_text']:
            if st.button("🎵 Convert to Audio", type="primary"):
                input_text = st.session_state['input_text']
                word_count = len(input_text.split())

                if word_count > 10000:
                    st.warning(f"⚠️ Your text has {word_count} words. Consider splitting into smaller chunks (max ~5000 words recommended).")

                with st.spinner("🎙️ Generating audio... This may take a moment."):
                    audio_data = generate_audio_tts(
                        text=input_text,
                        api_key=api_key,
                        voice_name=selected_voice,
                        speaking_style=speaking_style
                    )

                if audio_data:
                    audio_buffer = save_wave_file(audio_data)

                    st.markdown('<div class="success-box">✅ Audio generated successfully!</div>', unsafe_allow_html=True)

                    st.audio(audio_buffer, format='audio/wav')

                    timestamp = time.strftime("%Y%m%d-%H%M%S")
                    filename = f"audio_output_{timestamp}.wav"

                    st.download_button(
                        label="⬇️ Download Audio File",
                        data=audio_buffer,
                        file_name=filename,
                        mime="audio/wav"
                    )

                    st.info(f"🎵 Voice: {selected_voice}
📝 Words: {word_count}")

        else:
            st.info("👈 Please provide:")
            if not api_key:
                st.warning("🔑 Enter your Gemini API key in the sidebar")
            if 'input_text' not in st.session_state or not st.session_state.get('input_text'):
                st.warning("📝 Upload a file or type text in the left panel")

    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #666;'>
            <p>🚀 Powered by <strong>Google Gemini 2.5 Flash TTS</strong> | 
            Built with <strong>Streamlit</strong></p>
            <p style='font-size: 0.9em;'>
            ⚡ Get your free API key at <a href='https://ai.google.dev/' target='_blank'>ai.google.dev</a>
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()