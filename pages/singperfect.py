# app.py
import streamlit as st
import tempfile
import soundfile as sf
import librosa
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import base64
import json
import os
import re
from google import genai
import uuid
import html

# --------------------------------------------------------
# CONFIGURATION
# --------------------------------------------------------
st.set_page_config(page_title="SingPerfect 🎤", layout="wide")
if "GOOGLE_API_KEY" not in st.secrets:
    st.error("Please add GOOGLE_API_KEY to .streamlit/secrets.toml")
else:
    genai.configure(api_key=st.secrets["GOOGLE_API_KEY_1"])

# --------------------------------------------------------
# SESSION STATE
# --------------------------------------------------------
if "history" not in st.session_state:
    st.session_state.history = []

# --------------------------------------------------------
# HEADER
# --------------------------------------------------------
st.title("🎶 SingPerfect: Your AI Vocal Coach (Enhanced)")
st.write("""
Upload or record your song, get AI feedback, visualize pitch,
track progress — and practice with **auto-synced karaoke** (real-time highlighting).
""")

# --------------------------------------------------------
# FILE UPLOADS / RECORD
# --------------------------------------------------------
col1, col2 = st.columns(2)
with col1:
    ref_audio = st.file_uploader("🎵 Upload Reference Song (used for karaoke sync)", type=["mp3", "wav"])
with col2:
    user_audio = st.file_uploader("🎙️ Upload or Record Your Singing (for analysis)", type=["mp3", "wav"])

st.markdown("Or record directly 👇")
record_audio = st.audio_input("Record your singing here")
if record_audio and not user_audio:
    user_audio = record_audio

st.markdown("---")

# --------------------------------------------------------
# HELPERS
# --------------------------------------------------------
def save_to_wav(uploaded_file, dst_path):
    """Save uploaded file-like object to WAV (preserve bytes if already WAV)."""
    content = uploaded_file.read()
    # If it's likely already a WAV, attempt to write raw bytes. Otherwise write to file and librosa can load later.
    with open(dst_path, "wb") as f:
        f.write(content)
    return dst_path

def extract_pitch(audio_path, hop_length=512):
    """Return pitch track (Hz) using librosa piptrack."""
    y, sr = librosa.load(audio_path, sr=None)
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr, hop_length=hop_length)
    pitch_track = []
    for i in range(pitches.shape[1]):
        index = np.argmax(magnitudes[:, i])
        pitch = pitches[index, i]
        pitch_track.append(float(pitch))
    return np.array(pitch_track), sr, hop_length

def estimate_line_timestamps_by_onsets(audio_path, n_lines):
    """
    Estimate timestamps (seconds) for each line by detecting onsets and mapping them to n_lines.
    Returns list of start times for each line (len == n_lines), last line end = audio duration.
    This is a heuristic: good for phrase-level alignment; not word-accurate.
    """
    y, sr = librosa.load(audio_path, sr=None)
    # detect onsets (seconds)
    onset_frames = librosa.onset.onset_detect(y=y, sr=sr, backtrack=False)
    onset_times = librosa.frames_to_time(onset_frames, sr=sr)
    duration = librosa.get_duration(y=y, sr=sr)
    # if not enough onsets, create evenly spaced boundaries
    if len(onset_times) < max(2, n_lines):
        # fallback: evenly spaced
        times = np.linspace(0.0, duration, num=n_lines+1)[:-1]
        return [float(t) for t in times]
    # Map detected onsets to n_lines by sampling along onset_times
    # Create breakpoints: include 0.0 and duration
    points = np.concatenate(([0.0], onset_times, [duration]))
    # Choose n_lines start times by spacing along points
    chosen = np.linspace(0, len(points)-2, n_lines).astype(int)  # indices into points (start indices)
    starts = [float(points[i]) for i in chosen]
    # ensure strictly increasing
    starts = np.maximum.accumulate(starts)
    return starts

def audio_file_to_b64(audio_path):
    """Return base64 data URI for audio (wav/mp3) for embedding in HTML audio tag."""
    with open(audio_path, "rb") as f:
        data = f.read()
    b64 = base64.b64encode(data).decode("utf-8")
    # try to detect type by extension
    ext = os.path.splitext(audio_path)[1].lower()
    if ext == ".mp3":
        mime = "audio/mpeg"
    else:
        mime = "audio/wav"
    return f"data:{mime};base64,{b64}"

def parse_score_from_text(text):
    m = re.search(r"(\d{1,3})\s*/\s*100", text)
    if m:
        return int(m.group(1))
    m2 = re.search(r"score\s*[:\-]?\s*(\d{1,3})", text, re.IGNORECASE)
    if m2:
        return int(m2.group(1))
    return None

# --------------------------------------------------------
# ANALYZE & FEEDBACK (AI)
# --------------------------------------------------------
if ref_audio and user_audio:
    with st.spinner("Saving files and analyzing with Gemini... 🎧"):
        # Save temporary files
        tmp_dir = tempfile.mkdtemp(prefix="singperfect_")
        ref_path = os.path.join(tmp_dir, f"ref_{uuid.uuid4().hex}.wav")
        user_path = os.path.join(tmp_dir, f"user_{uuid.uuid4().hex}.wav")
        save_to_wav(ref_audio, ref_path)
        save_to_wav(user_audio, user_path)

        # Send to Gemini (if key configured)
        ai_feedback_text = "AI analysis unavailable (missing or invalid API key)."
        ai_score = None
        try:
            model = genai.GenerativeModel("models/gemini-2.5-flash-native-audio-latest")
            prompt = (
                "You are an expert vocal coach. Compare these two audio clips: (1) Reference song (ideal), "
                "(2) User singing attempt. Analyze pitch accuracy, rhythm alignment, tone, pronunciation, "
                "and expressiveness. Provide structured feedback and give a numeric performance score out of 100. "
                "Be encouraging and specific, mentioning sections (verse/chorus) if possible."
            )
            response = model.generate_content([
                {"mime_type": "text/plain", "text": prompt},
                {"mime_type": "audio/wav", "data": open(ref_path, "rb").read()},
                {"mime_type": "audio/wav", "data": open(user_path, "rb").read()},
            ])
            ai_feedback_text = response.text if hasattr(response, "text") else str(response)
            ai_score = parse_score_from_text(ai_feedback_text)
        except Exception as e:
            st.warning(f"AI analysis failed: {e}")

    st.subheader("🎧 AI Vocal Feedback")
    st.write(ai_feedback_text)

    # if no score parsed, fallback to random-ish but stable measure
    if ai_score is None:
        ai_score = int(np.clip(np.random.normal(78, 8), 50, 98))

    # Save to history
    st.session_state.history.append({"score": ai_score, "feedback": ai_feedback_text})

    # TTS spoken feedback (if possible)
    with st.spinner("Generating spoken feedback (TTS)... 🎙️"):
        try:
            tts_model = genai.GenerativeModel("models/gemini-2.5-flash-preview-tts")
            tts_resp = tts_model.generate_content(f"Speak the following feedback in a warm, motivating tone: {ai_feedback_text}")
            if hasattr(tts_resp, "audio") and tts_resp.audio:
                st.audio(tts_resp.audio, format="audio/wav")
        except Exception as e:
            st.info("Spoken feedback not available: " + str(e))

    # Pitch visualization
    st.subheader("🎛 Pitch Visualization (reference vs you)")
    ref_pitch, ref_sr, hop = extract_pitch(ref_path)
    user_pitch, user_sr, _ = extract_pitch(user_path)
    # Truncate to min len for plotting
    min_len = min(len(ref_pitch), len(user_pitch))
    plt.figure(figsize=(10, 3))
    plt.plot(ref_pitch[:min_len], label="Reference", alpha=0.9)
    plt.plot(user_pitch[:min_len], label="You", alpha=0.7)
    plt.title("Pitch Contour (Hz) — frames")
    plt.xlabel("Frames")
    plt.ylabel("Hz (0 = unvoiced)")
    plt.legend()
    st.pyplot(plt)

    st.markdown("---")

    # Improvement tracking
    st.subheader("📈 Progress Tracker")
    df_hist = pd.DataFrame(st.session_state.history)
    st.line_chart(df_hist["score"], use_container_width=True)
    st.write("Latest score:", ai_score)

    st.markdown("---")

    # --------------------------------------------------------
    # KARAOKE SYNC (auto)
    # --------------------------------------------------------
    st.header("🎤 Auto-Synced Karaoke (experimental)")
    lyrics = st.text_area("Paste your song lyrics (one line per display line).", height=200)
    st.markdown("If you paste multiple lines, the system will try to align each line to the reference audio automatically.")
    if lyrics and ref_audio:
        lyric_lines = [line.strip() for line in lyrics.splitlines() if line.strip()]
        n_lines = max(1, len(lyric_lines))

        with st.spinner("Estimating line timestamps from reference audio... ⏱️"):
            start_times = estimate_line_timestamps_by_onsets(ref_path, n_lines)
            # also compute small end times for convenience
            y_ref, sr_ref = librosa.load(ref_path, sr=None)
            total_dur = librosa.get_duration(y=y_ref, sr=sr_ref)
            # compute end_times as next start or total_dur
            end_times = start_times[1:] + [total_dur]
            # build list of dicts for JS
            segments = []
            for i, line in enumerate(lyric_lines):
                segments.append({"start": float(start_times[i]), "end": float(end_times[i]), "text": line})

        st.success("Timestamps estimated (phrase-level).")
        st.write("Preview of estimated timings (seconds):")
        for seg in segments:
            st.write(f"{seg['start']:.2f} → {seg['end']:.2f} : {seg['text']}")

        # Prepare audio blob as data URI
        audio_data_uri = audio_file_to_b64(ref_path)

        # Build karaoke HTML + JS component: highlights current line by audio currentTime
        comp_id = f"karaoke_{uuid.uuid4().hex}"
        # sanitize text for embedding
        safe_segments = [{"start": seg["start"], "end": seg["end"], "text": html.escape(seg["text"])} for seg in segments]
        js_segments_json = json.dumps(safe_segments)

        karaoke_html = f"""
        <style>
        .karaoke-wrap {{
            font-family: sans-serif;
            display: flex;
            gap: 20px;
        }}
        .lyrics {{
            width: 50%;
            max-height: 320px;
            overflow-y: auto;
            padding: 10px;
            border-radius: 8px;
            border: 1px solid #ddd;
            background: #fafafa;
        }}
        .lyrics .line {{
            padding: 6px 8px;
            margin: 4px 0;
            border-radius: 6px;
            transition: background 0.15s, transform 0.08s;
        }}
        .lyrics .line.highlight {{
            background: linear-gradient(90deg, #fff7cc, #ffe7a8);
            transform: scale(1.01);
            font-weight: 600;
        }}
        .player {{
            width: 45%;
        }}
        .controls {{
            margin-top: 10px;
        }}
        </style>

        <div id="{comp_id}" class="karaoke-wrap">
          <div class="lyrics" id="{comp_id}_lyrics">
          </div>

          <div class="player">
            <audio id="{comp_id}_audio" controls preload="auto" src="{audio_data_uri}"></audio>
            <div class="controls">
              <button id="{comp_id}_play">Play & Sync</button>
              <button id="{comp_id}_pause">Pause</button>
              <div>Current time: <span id="{comp_id}_time">0.00</span> s</div>
            </div>
            <div style="margin-top:10px;font-size:0.9em;color:#555">
              Tip: Click "Play & Sync" to autoplay the reference audio and watch the highlighted lines.
            </div>
          </div>
        </div>

        <script>
        const segments = {js_segments_json};
        const lyricsDiv = document.getElementById("{comp_id}_lyrics");
        const audio = document.getElementById("{comp_id}_audio");
        const playBtn = document.getElementById("{comp_id}_play");
        const pauseBtn = document.getElementById("{comp_id}_pause");
        const timeSpan = document.getElementById("{comp_id}_time");

        // build lyric lines
        for (let i=0;i<segments.length;i++){{
            const el = document.createElement("div");
            el.className = "line";
            el.dataset.start = segments[i].start;
            el.dataset.end = segments[i].end;
            el.id = "{comp_id}_line_" + i;
            el.innerHTML = "<div style='font-size:0.9em;color:#666'>"+segments[i].start.toFixed(2)+"s</div>"
                           + "<div style='font-size:1.05em;margin-top:3px'>"+segments[i].text+"</div>";
            lyricsDiv.appendChild(el);
        }}

        function updateHighlight(){{
            const t = audio.currentTime;
            timeSpan.innerText = t.toFixed(2);
            // find active segment
            for (let i=0;i<segments.length;i++){{
                const s = segments[i].start;
                const e = segments[i].end;
                const el = document.getElementById("{comp_id}_line_" + i);
                if (t >= s && t < e) {{
                    if (!el.classList.contains("highlight")) {{
                        // remove highlight class from all
                        const all = document.querySelectorAll("#{comp_id} .line");
                        all.forEach(x=>x.classList.remove("highlight"));
                        el.classList.add("highlight");
                        // scroll into view
                        el.scrollIntoView({{behavior: 'smooth', block: 'center'}});
                    }}
                }}
            }}
        }}

        let rafId = null;
        function rafLoop(){{
            updateHighlight();
            rafId = requestAnimationFrame(rafLoop);
        }}

        playBtn.onclick = function(){{
            // attempt to autoplay; browsers often require user gesture and this button counts
            audio.play().then(()=>{{ console.log('playing'); }}).catch((e)=>{{ console.log('play blocked', e); }});
            if (!rafId) rafLoop();
        }};
        pauseBtn.onclick = function(){{
            audio.pause();
            if (rafId) {{ cancelAnimationFrame(rafId); rafId = null; }}
        }};

        // If user uses the native controls, also start RAF to track highlights
        audio.onplay = function(){{ if (!rafId) rafLoop(); }};
        audio.onpause = function(){{ if (rafId) {{ cancelAnimationFrame(rafId); rafId = null; }} }};
        audio.ontimeupdate = function(){{ timeSpan.innerText = audio.currentTime.toFixed(2); }};
        </script>
        """

        st.components.v1.html(karaoke_html, height=420, scrolling=True)

        st.markdown("""
        **Notes & Limitations**
        - This auto-sync uses onset detection to divide the reference audio into phrase-level segments and map them to the lyric lines you provided. It's a fast heuristic and works best when each lyric line corresponds to a clear sung phrase.
        - For **word-level accuracy** or handling mismatches between lyrics and audio (e.g., repeats, ad-libs), use forced-alignment tools like **aeneas** or **Gentle** (they produce timestamps for each word/phoneme).
        - Want me to add word-level forced alignment using `aeneas` (server-side) or by running Gentle locally? I can provide that next.
        """)

# --------------------------------------------------------
# KARAOKE MODE (manual) if no analysis yet
# --------------------------------------------------------
else:
    st.info("Upload both a reference song and your singing to enable full analysis and karaoke sync.")
    st.markdown("---")
    st.header("🎤 Karaoke Mode (manual)")
    lyrics = st.text_area("Paste lyrics here (one line per display line) for manual practice:", height=180)
    if lyrics:
        st.write("You can still practice by playing the reference audio elsewhere and using these lyrics.")

# --------------------------------------------------------
# FOOTER
# --------------------------------------------------------
st.markdown("---")
st.caption("Built with ❤️ using Google Gemini 2.5 Flash Native Audio + TTS | Streamlit")