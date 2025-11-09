import streamlit as st
import tempfile
import numpy as np
import soundfile as sf
from pydub import AudioSegment
import matplotlib.pyplot as plt
from google import genai
from google.genai import types
from streamlit.components.v1 import html
import base64

# ===============================
# STREAMLIT PAGE SETTINGS
# ===============================
st.set_page_config(page_title="🎙️ AI Vocal Coach", layout="wide")

st.markdown("""
<style>
#MainMenu, footer, header {visibility: hidden;}
.main {padding: 2rem;}
</style>
""", unsafe_allow_html=True)

# ===============================
# HELPER FUNCTIONS
# ===============================
def safe_read_audio(path):
    try:
        y, sr = sf.read(path)
        if y.ndim > 1:
            y = np.mean(y, axis=1)
        return y, sr
    except Exception:
        audio = AudioSegment.from_file(path)
        y = np.array(audio.get_array_of_samples()).astype(float)
        sr = audio.frame_rate
        return y, sr


def load_audio_energy(path):
    y, sr = safe_read_audio(path)
    frame_len = int(0.05 * sr)
    hop = int(0.025 * sr)
    energies = []
    for i in range(0, len(y) - frame_len, hop):
        energies.append(np.mean(np.abs(y[i:i + frame_len])))
    energies = np.array(energies)
    return energies / np.max(energies) if np.max(energies) != 0 else energies


# ===============================
# GEMINI CLIENT
# ===============================
@st.cache_resource
def get_client():
    if "GOOGLE_API_KEY_1" not in st.secrets:
        st.error("Missing GOOGLE_API_KEY_1 in secrets.")
        st.stop()
    return genai.Client(api_key=st.secrets["GOOGLE_API_KEY"])

client = get_client()

# ===============================
# 1️⃣ UPLOAD REFERENCE SONG
# ===============================
st.header("🎧 Step 1: Upload Reference Song")

ref_file = st.file_uploader("Upload a song (mp3 or wav)", type=["mp3", "wav"])
ref_tmp_path = None

if ref_file:
    ref_tmp_path = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name
    with open(ref_tmp_path, "wb") as f:
        f.write(ref_file.read())

    # Extract lyrics only once
    if "lyrics_text" not in st.session_state:
        st.session_state["lyrics_text"] = ""
        with st.spinner("🎶 Extracting lyrics..."):
            try:
                response = client.models.generate_content(
                    model="gemini-2.5-pro",
                    contents=[
                        {
                            "role": "user",
                            "parts": [
                                {"text": "Extract full lyrics from this song and return only the lyrics text."},
                                {
                                    "inline_data": {
                                        "mime_type": "audio/wav",
                                        "data": open(ref_tmp_path, "rb").read(),
                                    }
                                },
                            ],
                        }
                    ],
                )
                st.session_state["lyrics_text"] = response.candidates[0].content.parts[0].text.strip()
            except Exception as e:
                st.warning(f"Lyrics extraction failed: {e}")

    st.audio(ref_tmp_path, format="audio/wav")
    st.subheader("📜 Lyrics (Sing Along)")

    lyrics_text = st.session_state["lyrics_text"]
    if lyrics_text:
        # Build karaoke with timed highlighting
        audio = AudioSegment.from_file(ref_tmp_path)
        duration_sec = audio.duration_seconds
        lines = [line.strip() for line in lyrics_text.split("\n") if line.strip()]
        n_lines = len(lines)
        timestamps = [round(i * (duration_sec / n_lines), 2) for i in range(n_lines)]

        lines_html = "".join(
            [f'<p class="lyric-line" data-time="{timestamps[i]}">{lines[i]}</p>' for i in range(n_lines)]
        )

        karaoke_html = f"""
        <div>
            <audio id="karaokePlayer" controls style="width:100%;">
                <source src="data:audio/wav;base64,{base64.b64encode(open(ref_tmp_path,'rb').read()).decode()}" type="audio/wav">
            </audio>
            <div id="lyrics-box" style="height:350px;overflow-y:auto;border:1px solid #ccc;
                padding:10px;font-size:1.1rem;line-height:1.6;margin-top:10px;">
                {lines_html}
            </div>
        </div>
        <script>
        const audio = document.getElementById('karaokePlayer');
        const lines = Array.from(document.querySelectorAll('.lyric-line'));
        const times = lines.map(l => parseFloat(l.dataset.time));
        let active = 0;
        function highlight(time) {{
            for (let i=0;i<lines.length;i++) {{
                if (time >= times[i] && (i===lines.length-1 || time < times[i+1])) {{
                    if (active!==i) {{
                        lines.forEach(l=>l.style.color='#444');
                        lines[i].style.color='#ff4081';
                        lines[i].scrollIntoView({{behavior:'smooth', block:'center'}});
                        active=i;
                    }}
                    break;
                }}
            }}
        }}
        audio.addEventListener('timeupdate',()=>highlight(audio.currentTime));
        audio.addEventListener('seeked',()=>highlight(audio.currentTime));
        </script>
        """
        html(karaoke_html, height=420)
    else:
        st.info("Lyrics not available.")


# ===============================
# 2️⃣ RECORD USER SINGING
# ===============================
st.header("🎤 Step 2: Record Your Singing")
recorded_audio = st.audio_input("🎙️ Record your singing")

recorded_path = None
if recorded_audio:
    recorded_path = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name
    with open(recorded_path, "wb") as f:
        f.write(recorded_audio.getvalue())
    st.success("✅ Recorded successfully!")

# ===============================
# 3️⃣ SIDE-BY-SIDE PLAYERS
# ===============================
if ref_tmp_path and recorded_path:
    st.header("🎧 Step 3: Compare Recordings")

    col1, col2 = st.columns(2)
    with col1:
        st.audio(ref_tmp_path)
        st.caption("Reference Song")

    with col2:
        st.audio(recorded_path)
        st.caption("Your Recording")

    # Energy comparison
    st.subheader("📊 Energy Comparison")
    ref_energy = load_audio_energy(ref_tmp_path)
    user_energy = load_audio_energy(recorded_path)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(ref_energy, label="Reference", linewidth=2)
    ax.plot(user_energy, label="Your Singing", linewidth=2)
    ax.legend()
    ax.set_xlabel("Frame")
    ax.set_ylabel("Normalized Energy")
    ax.set_title("Energy Contour Comparison")
    st.pyplot(fig)

    # ===============================
    # 4️⃣ VOCAL FEEDBACK
    # ===============================
    st.header("🎶 AI Vocal Feedback")
    with st.spinner("Analyzing vocals..."):
        try:
            feedback_prompt = (
                "You are a kind vocal coach. Compare user's recording to the reference song. "
                "Give short feedback about pitch, rhythm, and tone."
            )
            response = client.models.generate_content(
                model="gemini-2.5-pro",
                contents=[
                    {
                        "role": "user",
                        "parts": [
                            {"text": feedback_prompt},
                            {
                                "inline_data": {
                                    "mime_type": "audio/wav",
                                    "data": open(ref_tmp_path, "rb").read(),
                                }
                            },
                            {
                                "inline_data": {
                                    "mime_type": "audio/wav",
                                    "data": open(recorded_path, "rb").read(),
                                }
                            },
                        ],
                    }
                ],
            )
            feedback_text = response.candidates[0].content.parts[0].text
            st.success("✅ Feedback ready!")
            st.write(feedback_text)
        except Exception as e:
            st.warning(f"Feedback generation failed: {e}")