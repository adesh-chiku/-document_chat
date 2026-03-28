# ==============================
# 🚀 SmartDoc + SRE AI (NO CONFIDENCE VERSION)
# ==============================

import streamlit as st
import pdfplumber
import docx
from openai import OpenAI
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from rank_bm25 import BM25Okapi

# 🔑 API KEY
API_KEY = st.secrets["API_KEY"]

client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=API_KEY
)

# ------------------------------
# 🎨 PAGE CONFIG
# ------------------------------
st.set_page_config(page_title="Smart AI Suite", layout="wide")

# ------------------------------
# 🌙 THEME
# ------------------------------
if "theme" not in st.session_state:
    st.session_state.theme = True

if "messages" not in st.session_state:
    st.session_state.messages = []

# ------------------------------
# 📌 SIDEBAR
# ------------------------------
with st.sidebar:
    st.title("🤖 Smart AI Suite")

    mode = st.selectbox(
        "Answer Style",
        ["Normal", "Explain Like I'm 5", "Detailed", "Bullet Points"]
    )

    show_sources = st.checkbox("Show Sources", True)

    theme_mode = st.toggle("🌙 Dark Mode", value=st.session_state.theme)
    st.session_state.theme = theme_mode

# ------------------------------
# 🎨 APPLY THEME
# ------------------------------
if st.session_state.theme:
    st.markdown(
        """
        <style>
        .stApp {background:#0E1117; color:white;}
        .stMarkdown, .stTextInput, .stTextArea, .stSelectbox, .stCheckbox, .stExpander, .stChatMessage {
            color: white !important;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
else:
    st.markdown(
        """
        <style>
        .stApp {background:white; color:black;}
        </style>
        """,
        unsafe_allow_html=True
    )

# ------------------------------
# 📄 FILE READER
# ------------------------------
def extract_text(file):
    name = file.name.lower()

    if name.endswith(".pdf"):
        text = ""
        with pdfplumber.open(file) as pdf:
            for page in pdf.pages:
                text += page.extract_text() or ""
        return text

    elif name.endswith(".docx"):
        doc = docx.Document(file)
        return " ".join([p.text for p in doc.paragraphs])

    elif name.endswith(".txt"):
        return file.read().decode("utf-8", errors="ignore")

# ------------------------------
# ✂️ TEXT SPLIT
# ------------------------------
def split_text(text, chunk_size=500, overlap=100):
    chunks = []
    for i in range(0, len(text), chunk_size - overlap):
        chunks.append(text[i:i+chunk_size])
    return chunks

# ------------------------------
# 🧠 MODEL
# ------------------------------
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()

# ------------------------------
# 🔍 VECTOR STORE
# ------------------------------
def create_vector_store(chunks):
    texts = [c["text"] for c in chunks]

    embeddings = model.encode(texts)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings))

    tokenized = [t.split() for t in texts]
    bm25 = BM25Okapi(tokenized)

    return index, embeddings, bm25

# ------------------------------
# 🔎 HYBRID SEARCH
# ------------------------------
def hybrid_search(query, chunks, index, embeddings, bm25, k=3):
    if "name" in query.lower() or "resume" in query.lower():
        k = 5

    query_embedding = model.encode([query])
    distances, indices = index.search(np.array(query_embedding), k)

    semantic = [chunks[i] for i in indices[0]]

    scores = bm25.get_scores(query.split())
    keyword_idx = np.argsort(scores)[-k:]
    keyword = [chunks[i] for i in keyword_idx]

    combined = semantic + keyword

    seen = set()
    unique = []
    for c in combined:
        if c["text"] not in seen:
            unique.append(c)
            seen.add(c["text"])

    return unique, distances

# ------------------------------
# 🤖 STREAMING
# ------------------------------
def stream_answer(prompt):
    response = client.chat.completions.create(
        model="meta/llama-3.1-70b-instruct",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=400,
        stream=True
    )

    for chunk in response:
        if hasattr(chunk.choices[0].delta, "content"):
            content = chunk.choices[0].delta.content
            if content:
                yield content

# ------------------------------
# 🤖 SIMPLE AI
# ------------------------------
def ask_ai(prompt):
    response = client.chat.completions.create(
        model="meta/llama-3.1-70b-instruct",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=400
    )
    return response.choices[0].message.content

# ------------------------------
# 🎯 STYLE
# ------------------------------
style_map = {
    "Normal": "",
    "Explain Like I'm 5": "Explain like a 5-year-old.",
    "Detailed": "Give detailed explanation.",
    "Bullet Points": "Answer in bullet points."
}

# ------------------------------
# 📂 TABS
# ------------------------------
tabs = st.tabs([
    "📄 Document Chat",
    "🔍 Log Analyzer",
    "📊 Health Check",
    "📚 Runbook",
    "🚨 Alert"
])

# ------------------------------
# 📄 DOCUMENT CHAT
# ------------------------------
with tabs[0]:
    files = st.file_uploader("Upload Documents", accept_multiple_files=True)

    # Show history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if files:
        chunks = []

        for file in files:
            text = extract_text(file)

            for chunk in split_text(text):
                chunks.append({
                    "text": chunk,
                    "source": file.name
                })

        st.success("Documents processed ✅")

        index, embeddings, bm25 = create_vector_store(chunks)

        query = st.chat_input("Ask your question...")

        if query:
            st.session_state.messages.append({"role": "user", "content": query})

            with st.chat_message("user"):
                st.markdown(query)

            results, distances = hybrid_search(query, chunks, index, embeddings, bm25)

            context = " ".join([r["text"] for r in results])

            if "name" in query.lower() or "resume" in query.lower():
                prompt = f"Find the NAME of the person from the context.\nContext: {context}\nQuestion: {query}"
            else:
                prompt = f"{style_map[mode]}\nContext: {context}\nQuestion: {query}"

            with st.chat_message("assistant"):
                placeholder = st.empty()
                full = ""

                for chunk in stream_answer(prompt):
                    full += chunk
                    placeholder.markdown(full + "▌")

                placeholder.markdown(full)

            st.session_state.messages.append({"role": "assistant", "content": full})

            if show_sources:
                with st.expander("📄 Sources"):
                    for r in results:
                        st.write(f"📁 {r['source']}")
                        st.write(r["text"])
                        st.write("---")

# ------------------------------
# 🔍 LOG ANALYZER
# ------------------------------
with tabs[1]:
    logs = st.text_area("Paste Logs")

    if st.button("Analyze Logs"):
        st.write(ask_ai(f"Analyze logs:\n{logs}"))

# ------------------------------
# 📊 HEALTH CHECK
# ------------------------------
with tabs[2]:
    metrics = st.text_area("Enter Metrics")

    if st.button("Check Health"):
        st.write(ask_ai(f"Analyze system metrics:\n{metrics}"))

# ------------------------------
# 📚 RUNBOOK
# ------------------------------
with tabs[3]:
    issue = st.text_input("Enter Issue")

    if st.button("Generate Runbook"):
        st.write(ask_ai(f"Create runbook for:\n{issue}"))

# ------------------------------
# 🚨 ALERT
# ------------------------------
with tabs[4]:
    alert = st.text_area("Paste Alert")

    if st.button("Explain Alert"):
        st.write(ask_ai(f"Explain alert:\n{alert}"))
