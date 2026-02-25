import google.generativeai as genai
import streamlit as st
from pypdf import PdfReader


def init_gemini():
    genai.configure(api_key=st.secrets["KEY_11"])
    return genai


def get_embedding(genai, text):
    response = genai.embed_content(
        model="models/text-embedding-004",
        content=text
    )
    return response["embedding"]


def extract_pdf_text(file):
    reader = PdfReader(file)
    text = ""
    for page in reader.pages:
        if page.extract_text():
            text += page.extract_text() + "\n"
    return text