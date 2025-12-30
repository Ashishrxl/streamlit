import streamlit as st
import numpy as np
from utils import init_gemini, get_embedding, extract_pdf_text

# ---------------- CONFIG ---------------- #
st.set_page_config(
    page_title="🎓 AI Study Partner",
    layout="wide"
)

genai = init_gemini()
model = genai.GenerativeModel("models/gemini-2.5-flash")

# ---------------- SESSION STATE ---------------- #
if "notes" not in st.session_state:
    st.session_state.notes = []

if "embeddings" not in st.session_state:
    st.session_state.embeddings = []

# ---------------- FUNCTIONS ---------------- #
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def get_best_context(question):
    q_embedding = get_embedding(genai, question)
    similarities = [
        cosine_similarity(q_embedding, emb)
        for emb in st.session_state.embeddings
    ]
    best_index = int(np.argmax(similarities))
    return st.session_state.notes[best_index]

# ---------------- UI ---------------- #
st.title("🎓 AI Study Partner")
st.caption("Upload notes → Ask questions → Get smarter explanations")

st.divider()

# ---------------- UPLOAD NOTES ---------------- #
st.header("📂 Upload Study Notes")

uploaded_file = st.file_uploader(
    "Upload PDF notes",
    type=["pdf"]
)

if uploaded_file:
    with st.spinner("Reading and learning your notes..."):
        text = extract_pdf_text(uploaded_file)
        embedding = get_embedding(genai, text)

        st.session_state.notes.append(text)
        st.session_state.embeddings.append(embedding)

    st.success("Notes uploaded successfully!")

st.divider()

# ---------------- ASK QUESTIONS ---------------- #
st.header("❓ Ask a Question")

question = st.text_input("Type your question here")

if question and st.session_state.notes:
    with st.spinner("Thinking..."):
        context = get_best_context(question)

        prompt = f"""
You are a friendly AI study tutor.

Use ONLY the notes below to answer.
Explain clearly, step by step.

NOTES:
{context}

QUESTION:
{question}
"""

        response = model.generate_content(prompt)

    st.subheader("📘 Answer")
    st.write(response.text)

elif question:
    st.warning("Please upload notes first.")

st.divider()

# ---------------- RE-EXPLAIN ---------------- #
st.header("🔁 Explain Differently")

style = st.selectbox(
    "Choose explanation style",
    [
        "Beginner",
        "Like a 10 year old",
        "With real-world analogy",
        "Exam focused",
        "Short & simple"
    ]
)

if st.button("Re-explain Topic") and st.session_state.notes:
    with st.spinner("Re-explaining..."):
        explain_prompt = f"""
Re-explain the content below.

Style: {style}

CONTENT:
{st.session_state.notes[-1]}
"""
        explanation = model.generate_content(explain_prompt)

    st.subheader("📖 Explanation")
    st.write(explanation.text)

st.divider()

# ---------------- QUIZ ---------------- #
st.header("📝 Generate Quiz")

if st.button("Create Quiz") and st.session_state.notes:
    with st.spinner("Creating quiz..."):
        quiz_prompt = f"""
Create 5 quiz questions from the content below.
Mix easy and hard.
Provide answers at the end.

CONTENT:
{st.session_state.notes[-1]}
"""
        quiz = model.generate_content(quiz_prompt)

    st.subheader("🧠 Quiz")
    st.write(quiz.text)

st.divider()

st.caption("🚀 Built with Gemini + Streamlit (Free Tier Friendly)")