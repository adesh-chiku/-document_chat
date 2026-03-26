# ==============================
# 🚀 SmartDoc AI (Pro UI + Theme + Settings)
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
# 🎨 Page Config
# ------------------------------
st.set_page_config(
    page_title="SmartDoc AI",
    page_icon="🤖",
    layout="wide"
)

# ------------------------------
# 🌙 Theme State
# ------------------------------
if "theme" not in st.session_state:
    st.session_state.theme = True  # Default Dark

# ------------------------------
# 📌 Sidebar
# ------------------------------
with st.sidebar:
    st.title("🤖 SmartDoc AI")
    st.markdown("AI-powered document assistant")
    st.divider()

    mode = st.selectbox(
        "Answer Style",
        ["Normal", "Explain Like I'm 5", "Detailed", "Bullet Points"]
    )

    st.divider()

    with st.expander("⚙️ Settings"):
        show_confidence = st.checkbox("Show Confidence", value=True)
        show_sources = st.checkbox("Show Sources", value=True)
        theme_mode = st.toggle("🌙 Dark Mode", value=st.session_state.theme)

        st.session_state.theme = theme_mode

# ------------------------------
# 🎨 Apply Theme
# ------------------------------
if st.session_state.theme:
    # DARK MODE
    st.markdown("""
        <style>
        .stApp {
            background-color: #0E1117;
            color: #FAFAFA;
        }
        .stChatMessage {
            background-color: #1E1E1E !important;
        }
        </style>
    """, unsafe_allow_html=True)
else:
    # LIGHT MODE
    st.markdown("""
        <style>
        .stApp {
            background-color: #FFFFFF;
            color: #000000;
        }
        .stChatMessage {
            background-color: #F5F5F5 !important;
        }
        </style>
    """, unsafe_allow_html=True)

# ------------------------------
# 🧾 Header
# ------------------------------
st.markdown("""
# 🤖 SmartDoc AI  
### Chat with your documents like a pro 🚀
""")

# ------------------------------
# 📄 File Reader
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
# ✂️ Chunking
# ------------------------------
def split_text(text, chunk_size=500, overlap=100):
    chunks = []
    for i in range(0, len(text), chunk_size - overlap):
        chunks.append(text[i:i+chunk_size])
    return chunks

# ------------------------------
# 🧠 Embedding Model
# ------------------------------
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()

# ------------------------------
# 🔍 Vector Store
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
# 🔎 Hybrid Search
# ------------------------------
def hybrid_search(query, chunks, index, embeddings, bm25, k=3):
    texts = [c["text"] for c in chunks]

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
# 🤖 Streaming
# ------------------------------
def get_answer_stream(prompt):

    response = client.chat.completions.create(
        model="meta/llama-3.1-70b-instruct",
        messages=[
            {"role": "system", "content": "You are a helpful AI assistant."},
            {"role": "user", "content": prompt}
        ],
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
# 🎯 Style Mapping
# ------------------------------
style_map = {
    "Normal": "",
    "Explain Like I'm 5": "Explain like you are talking to a 5-year-old child using simple words and examples.",
    "Detailed": "Give a detailed explanation.",
    "Bullet Points": "Answer in bullet points."
}

# ------------------------------
# 📂 Upload
# ------------------------------
st.markdown("### 📂 Upload your documents")

files = st.file_uploader(
    "",
    type=["pdf", "docx", "txt"],
    accept_multiple_files=True
)

chunks = []

if files:
    for file in files:
        text = extract_text(file)

        for chunk in split_text(text):
            chunks.append({
                "text": chunk,
                "source": file.name
            })

    st.success("✅ Documents processed!")

    index, embeddings, bm25 = create_vector_store(chunks)

# ------------------------------
# 💬 Chat
# ------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_input = st.chat_input("Ask your question...")

if user_input and files:

    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.markdown(user_input)

    results, distances = hybrid_search(user_input, chunks, index, embeddings, bm25)

    context = " ".join([r["text"] for r in results])

    prompt = f"""
    Context:
    {context}

    Instruction:
    {style_map[mode]}

    Question:
    {user_input}
    """

    with st.chat_message("assistant"):
        placeholder = st.empty()
        full_response = ""

        with st.spinner("🤖 Thinking..."):
            stream = get_answer_stream(prompt)

            for chunk in stream:
                full_response += chunk
                placeholder.markdown(full_response + "▌")

        placeholder.markdown(full_response)

    st.session_state.messages.append({"role": "assistant", "content": full_response})

    # 🎯 Confidence
    if show_confidence:
        confidence = 1 / (1 + distances[0][0])
        st.markdown(f"Confidence: {round(confidence*100,2)}%")

    # 📄 Sources
    if show_sources:
        with st.expander("📄 View Sources"):
            for r in results:
                st.markdown(f"📁 {r['source']}")
                st.write(r["text"])
                st.write("---")
