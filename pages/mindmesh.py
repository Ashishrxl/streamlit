import os
import json
import requests
import streamlit as st
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from pyvis.network import Network
import tempfile
from streamlit.components.v1 import html

# --- Hide Streamlit UI elements ---
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

hide_streamlit_style = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
[data-testid="stStatusWidget"] {display: none;}
[data-testid="stToolbar"] {display: none;}
a[href^="https://github.com"] {display: none !important;}
a[href^="https://streamlit.io"] {display: none !important;}
header > div:nth-child(2) {
    display: none;
}
</style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)


# ---------------- CONFIG ----------------
# Your Google Generative Language API key
GEMINI_API_KEY = st.secrets["GOOGLE_API_KEY_1"]
if not GEMINI_API_KEY:
    st.error("❌ Missing GEMINI_API_KEY. Please add it to Streamlit secrets.")
    st.stop()

# ✅ Latest verified Gemini endpoints and models (valid in India & globally)
BASE_GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta"
GEN_MODEL = "gemini-2.5-pro"
EMB_MODEL = "embedding-001"


# ---------------- HELPERS ----------------
def gemini_generate(prompt, temperature=0.4, max_output_tokens=512):
    """Generate a natural language reply using Gemini 2.5 Pro with robust parsing."""
    url = f"{BASE_GEMINI_URL}/models/{GEN_MODEL}:generateContent?key={GEMINI_API_KEY}"
    body = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": temperature,
            "maxOutputTokens": max_output_tokens
        }
    }

    r = requests.post(url, json=body, timeout=60)
    if r.status_code != 200:
        st.error(f"❌ Gemini API error: {r.status_code} — {r.text}")
        st.stop()

    data = r.json()

    # ✅ More robust parsing logic
    try:
        candidates = data.get("candidates", [])
        if candidates:
            content = candidates[0].get("content", {})
            parts = content.get("parts", [])
            if parts and "text" in parts[0]:
                return parts[0]["text"].strip()
        # fallback: try "outputText" if available
        if "outputText" in data:
            return data["outputText"].strip()
    except Exception as e:
        st.warning(f"⚠️ Parsing error: {e}")

    # if nothing parsed, show full JSON for debugging
    return json.dumps(data, indent=2)


def gemini_extract_concepts(conversation_text):
    """Ask Gemini to extract a semantic graph from conversation text."""
    extraction_prompt = f"""
Extract a semantic network of concepts from this conversation as valid JSON.

Conversation:
\"\"\"{conversation_text}\"\"\"

Return ONLY JSON with:
{{
  "nodes": [{{"id":"n1","label":"<concept>"}}],
  "edges": [{{"source":"n1","target":"n2","label":"relation"}}]
}}
Keep ids short. Avoid duplicates.
"""
    resp = gemini_generate(extraction_prompt, temperature=0.0, max_output_tokens=400)
    try:
        start, end = resp.find("{"), resp.rfind("}")
        if start != -1 and end != -1:
            json_text = resp[start:end + 1]
            return json.loads(json_text)
    except Exception:
        st.warning("⚠️ Could not parse JSON from Gemini.")
        st.text(resp)
    return {"nodes": [], "edges": []}


def gemini_embed(texts):
    """Generate embeddings using embedding-001 model."""
    vectors = []
    for text in texts:
        url = f"{BASE_GEMINI_URL}/models/{EMB_MODEL}:embedContent?key={GEMINI_API_KEY}"
        body = {
            "model": f"models/{EMB_MODEL}",
            "content": {"parts": [{"text": text}]}
        }
        r = requests.post(url, json=body, timeout=60)
        if r.status_code != 200:
            st.error(f"❌ Embedding API error: {r.status_code} — {r.text}")
            st.stop()
        data = r.json()
        emb = data.get("embedding", {}).get("values", [])
        vectors.append(np.array(emb, dtype=float))
    return vectors


# ---------------- SESSION STATE ----------------
if "nodes" not in st.session_state:
    st.session_state.nodes = []     # list of dicts {id,label}
if "edges" not in st.session_state:
    st.session_state.edges = []     # list of dicts {source,target,label}
if "vectors" not in st.session_state:
    st.session_state.vectors = []   # list of np arrays
if "conversations" not in st.session_state:
    st.session_state.conversations = []  # list of {role,text}


# ---------------- UI ----------------
st.set_page_config(page_title="MindMesh – Conversational Knowledge Mapper", layout="wide")
st.title("🧠 MindMesh — Conversational Knowledge Mapper")
st.caption("Talk with Gemini and see your ideas form a living mind map.")

col1, col2 = st.columns([1, 1])

# ==== LEFT: Chat ====
with col1:
    st.header("💬 Chat")

    user_input = st.text_input("You:", key="user_input")
    if st.button("Send"):
        if not user_input.strip():
            st.warning("Please enter a message.")
        else:
            st.session_state.conversations.append({"role": "user", "text": user_input})
            convo_text = "\n".join(
                [f"{c['role']}: {c['text']}" for c in st.session_state.conversations]
            )
            with st.spinner("Thinking..."):
                assistant_text = gemini_generate(convo_text)
            st.session_state.conversations.append({"role": "assistant", "text": assistant_text})
            st.rerun()

    st.markdown("### Conversation")
    for turn in st.session_state.conversations:
        if turn["role"] == "user":
            st.markdown(f"**You:** {turn['text']}")
        else:
            st.markdown(f"**Gemini:** {turn['text']}")

    if st.button("🧩 Extract Concepts & Update Graph"):
        convo_text = "\n".join([f"{c['role']}: {c['text']}" for c in st.session_state.conversations])
        with st.spinner("Extracting concepts..."):
            parsed = gemini_extract_concepts(convo_text)
        nodes = parsed.get("nodes", [])
        edges = parsed.get("edges", [])
        existing_labels = {n["label"]: n["id"] for n in st.session_state.nodes}
        new_labels = []
        for n in nodes:
            label = n.get("label")
            if label and label not in existing_labels:
                st.session_state.nodes.append(n)
                existing_labels[label] = n["id"]
                new_labels.append(label)
        if new_labels:
            with st.spinner("Embedding new concepts..."):
                embs = gemini_embed(new_labels)
            for e in embs:
                st.session_state.vectors.append(e)
        st.session_state.edges.extend(edges)
        st.success("✅ Graph updated!")


# ==== RIGHT: Graph ====
with col2:
    st.header("🧠 Knowledge Graph")

    def draw_pyvis(nodes, edges):
        g = Network(height="650px", width="100%", bgcolor="#ffffff", font_color="black")
        for n in nodes:
            g.add_node(n["id"], label=n["label"], title=n["label"], shape="dot", size=18)
        for e in edges:
            src, tgt = e.get("source"), e.get("target")
            if src and tgt:
                g.add_edge(src, tgt, title=e.get("label", ""), label=e.get("label", ""))
        path = tempfile.NamedTemporaryFile(suffix=".html", delete=False).name
        g.write_html(path)
        return path

    if st.session_state.nodes:
        html_path = draw_pyvis(st.session_state.nodes, st.session_state.edges)
        with open(html_path, "r", encoding="utf-8") as f:
            html_content = f.read()
        st.components.v1.html(html_content, height=700, scrolling=True)
    else:
        st.info("No concepts yet. Start chatting and extract concepts to see your graph!")

    st.markdown("### 🔍 Semantic Search")
    query = st.text_input("Find similar concepts:")
    if st.button("Search"):
        if not st.session_state.vectors:
            st.warning("No embeddings yet — extract concepts first.")
        elif not query.strip():
            st.warning("Enter a search phrase.")
        else:
            q_emb = gemini_embed([query])[0]
            vs = np.stack(st.session_state.vectors)
            sims = cosine_similarity([q_emb], vs)[0]
            idxs = np.argsort(-sims)[:5]
            st.markdown("**Top related concepts:**")
            for i in idxs:
                st.markdown(f"- {st.session_state.nodes[i]['label']}  —  *(similarity {sims[i]:.3f})*")

st.markdown("---")
st.caption("MindMesh • Built with Gemini 2.5 Pro + Streamlit • Keep your API key safe in Streamlit Secrets.")