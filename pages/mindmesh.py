# app.py
import os
import json
import requests
import streamlit as st
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
from pyvis.network import Network
import tempfile

# ------------ Configuration & helpers ------------
GEMINI_API_KEY = st.secrets["GOOGLE_API_KEY_1"]  # Put in Streamlit Cloud secrets
GEMINI_MODEL = "models/gemini-2.5-pro"        # generation model
EMBED_MODEL = "models/gemini-embedding-001"  # embedding model

BASE_GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta2"

def gemini_generate(prompt, temperature=0.2, max_output_tokens=512):
    """
    Calls the Gemini generate endpoint (REST). Returns generated text.
    """
    assert GEMINI_API_KEY, "GEMINI_API_KEY not set in environment."
    url = f"{BASE_GEMINI_URL}/{GEMINI_MODEL}:generate?key={GEMINI_API_KEY}"
    body = {
        "prompt": {
            "messages": [
                {"role": "system", "content": {"text": "You are a helpful assistant that returns natural language."}},
                {"role": "user", "content": {"text": prompt}}
            ]
        },
        "temperature": temperature,
        "maxOutputTokens": max_output_tokens,
    }
    r = requests.post(url, json=body, timeout=60)
    r.raise_for_status()
    data = r.json()
    # The text response path may vary by API version; this extracts the generated text.
    # Safe access:
    generations = data.get("candidates", [])
    if len(generations) > 0:
        return generations[0].get("output", {}).get("content", [{"text": ""}])[0].get("text", "")
    # Fallback
    return data.get("output", {}).get("text", "")

def gemini_extract_concepts(conversation_text):
    """
    Ask Gemini to extract a compact JSON of concepts and relations from the conversation_text.
    We instruct Gemini to output strictly JSON: {"nodes": [{"id": "...","label":"..."}], "edges":[{"source":"id","target":"id","label":"..."}]}
    """
    extraction_prompt = f"""
You are an assistant that extracts the core semantic graph from a conversation, producing ONLY valid JSON.
Input conversation:
\"\"\"{conversation_text}\"\"\"

Return a JSON object with two arrays: "nodes" and "edges".
- nodes: each node is {{ "id": "<short-id>", "label": "<concept string>" }}
- edges: each edge is {{ "source": "<id>", "target": "<id>", "label": "<relationship short label (optional)>" }}

Only output JSON. Use short ids (like n1, n2). Keep concepts concise.
"""
    resp = gemini_generate(extraction_prompt, temperature=0.0, max_output_tokens=400)
    # Try to find JSON substring (Gemini might append whitespace)
    # We'll attempt to parse the first JSON object in resp
    try:
        # find the first '{' and last '}' to be robust
        start = resp.find("{")
        end = resp.rfind("}")
        if start != -1 and end != -1:
            json_text = resp[start:end+1]
            parsed = json.loads(json_text)
            return parsed
    except Exception as e:
        st.warning("Failed to parse concept JSON from Gemini. Raw extraction output shown in logs.")
        st.write(resp)
    return {"nodes": [], "edges": []}

def gemini_embed(texts):
    """
    Call Gemini embeddings endpoint with a list of strings and return list of vectors (lists).
    """
    assert GEMINI_API_KEY, "GEMINI_API_KEY not set."
    url = f"{BASE_GEMINI_URL}/{EMBED_MODEL}:embed?key={GEMINI_API_KEY}"
    body = {"input": texts}
    r = requests.post(url, json=body, timeout=60)
    r.raise_for_status()
    data = r.json()
    # Data format: {'embeddings': [{'embedding': [..]}, ...]} or similar
    embs = []
    if "embeddings" in data:
        for e in data["embeddings"]:
            embs.append(np.array(e["embedding"], dtype=float))
    else:
        # fallback if different shape
        items = data.get("responses") or []
        for it in items:
            emb = it.get("embedding") or it.get("embeddings")
            if emb:
                embs.append(np.array(emb, dtype=float))
    return embs

# Simple in-memory vector store (persist only while app runs)
if "nodes" not in st.session_state:
    st.session_state["nodes"] = []     # list of dicts {id,label}
if "edges" not in st.session_state:
    st.session_state["edges"] = []     # list of dicts {source,target,label}
if "vectors" not in st.session_state:
    st.session_state["vectors"] = []   # list of np arrays aligned with nodes
if "conversations" not in st.session_state:
    st.session_state["conversations"] = []  # list of (user, assistant) pairs

# ------------ Streamlit UI ------------
st.set_page_config(page_title="MindMesh — Conversational Knowledge Mapper", layout="wide")
st.title("🧠 MindMesh — Conversational Knowledge Mapper")
st.markdown("Have a deep conversation with Gemini and build a visual knowledge graph of the ideas discussed.")

col1, col2 = st.columns([1, 1])

with col1:
    st.header("Chat")
    user_input = st.text_input("You:", key="user_input")
    if st.button("Send"):
        if user_input.strip() == "":
            st.warning("Please write something.")
        else:
            # append user turn
            st.session_state.conversations.append({"role":"user","text":user_input})
            # Build conversation text for context (last N turns)
            convo_text = "\n".join([f"{c['role']}: {c['text']}" for c in st.session_state.conversations])
            # Get assistant reply
            with st.spinner("Generating reply..."):
                assistant_text = gemini_generate(convo_text)
            st.session_state.conversations.append({"role":"assistant","text":assistant_text})
            st.success("Assistant replied.")
            st.experimental_rerun()

    # Show conversation
    st.markdown("**Conversation**")
    for turn in st.session_state.conversations:
        role = turn["role"]
        txt = turn["text"]
        if role == "user":
            st.markdown(f"**You:** {txt}")
        else:
            st.markdown(f"**Assistant:** {txt}")

    # Button to extract concepts & update graph
    if st.button("Extract concepts → Update graph"):
        convo_text = "\n".join([f"{c['role']}: {c['text']}" for c in st.session_state.conversations])
        with st.spinner("Extracting concepts (Gemini) ..."):
            parsed = gemini_extract_concepts(convo_text)
        nodes = parsed.get("nodes", [])
        edges = parsed.get("edges", [])
        # Add nodes (avoid duplicates by label)
        existing_labels = {n["label"]: n["id"] for n in st.session_state.nodes}
        new_nodes = []
        for n in nodes:
            label = n.get("label")
            nid = n.get("id")
            if label and label not in existing_labels:
                new_nodes.append(label)
                st.session_state.nodes.append({"id": nid, "label": label})
                existing_labels[label] = nid
        # generate embeddings for new_nodes and append to vectors
        if new_nodes:
            with st.spinner("Embedding new concepts..."):
                embs = gemini_embed(new_nodes)
            for e in embs:
                st.session_state.vectors.append(e)
        # add edges (attempt to map source/target ids to existing nodes)
        for ed in edges:
            st.session_state.edges.append(ed)
        st.success("Graph updated.")

with col2:
    st.header("Graph / Map")
    st.markdown("The semantic graph built from extracted concepts. Drag nodes to explore.")

    def draw_pyvis(nodes, edges, vectors):
        G = Network(height="650px", width="100%", bgcolor="#ffffff", font_color="black")
        # add nodes
        for n in nodes:
            G.add_node(n["id"], label=n["label"], title=n["label"], shape="dot", size=20)
        # add edges
        for e in edges:
            src = e.get("source")
            tgt = e.get("target")
            lab = e.get("label", "")
            if src and tgt:
                G.add_edge(src, tgt, title=lab, label=lab)
        # Save to temp and return HTML
        path = tempfile.NamedTemporaryFile(suffix=".html", delete=False).name
        G.show(path)
        return path

    graph_html_path = draw_pyvis(st.session_state.nodes, st.session_state.edges, st.session_state.vectors)
    # Show HTML
    with open(graph_html_path, "r", encoding="utf-8") as f:
        html = f.read()
    st.components.v1.html(html, height=700, scrolling=True)

    st.markdown("**Nearest concepts (semantic search):**")
    query = st.text_input("Find concepts similar to:", key="semantic_query")
    if st.button("Search similar"):
        if query.strip() == "":
            st.warning("Enter a query.")
        elif len(st.session_state.vectors) == 0:
            st.info("No concept vectors yet — extract concepts first.")
        else:
            q_emb = gemini_embed([query])[0]
            vs = np.stack(st.session_state.vectors)
            sims = cosine_similarity([q_emb], vs)[0]
            # top 5
            idxs = np.argsort(-sims)[:5]
            for i in idxs:
                node = st.session_state.nodes[i]
                st.markdown(f"- **{node['label']}** — similarity {sims[i]:.3f}")

st.markdown("---")
st.caption("Tip: store your GEMINI_API_KEY in Streamlit Cloud's Secrets and set it as an env var named GEMINI_API_KEY.")