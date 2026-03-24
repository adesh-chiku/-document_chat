# ==============================
# 🚀 Smart Chat with Documents (RAG + Streaming + Memory)
# ==============================

import streamlit as st
import pdfplumber
import docx
from openai import OpenAI
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# 🔑 API KEY
API_KEY = "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"  # apni key daalo

client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=API_KEY
)

# ------------------------------
# 📄 File readers
# ------------------------------
def extract_text_from_pdf(file):
    text = ""
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            text += page.extract_text() or ""
    return text

def extract_text_from_docx(file):
    doc = docx.Document(file)
    return " ".join([p.text for p in doc.paragraphs])

def extract_text_from_txt(file):
    return file.read().decode("utf-8", errors="ignore")

# ------------------------------
# ✂️ Split text
# ------------------------------
def split_text(text, chunk_size=500):
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

# ------------------------------
# 🧠 Embedding model
# ------------------------------
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()

# ------------------------------
# 🔍 Create vector store
# ------------------------------
def create_vector_store(chunks):
    embeddings = model.encode(chunks)
    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings))

    return index, embeddings

# ------------------------------
# 🔎 Search relevant chunks
# ------------------------------
def search_chunks(query, chunks, index, embeddings, k=3):
    query_embedding = model.encode([query])
    distances, indices = index.search(np.array(query_embedding), k)

    return [chunks[i] for i in indices[0]]

# ------------------------------
# 🤖 Streaming Answer
# ------------------------------
def get_answer_stream(context, question):

    prompt = f"""
    Answer based on context.

    Context:
    {context}

    Question:
    {question}
    """

    response = client.chat.completions.create(
        model="meta/llama-3.1-70b-instruct",
        messages=[
            {"role": "system", "content": "You are a helpful AI assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3,
        max_tokens=300,
        stream=True
    )

    for chunk in response:
        if hasattr(chunk.choices[0].delta, "content") and chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content

# ------------------------------
# 🖥️ UI
# ------------------------------
st.set_page_config(page_title="Smart Doc Chat", layout="wide")

st.title("💬 Chat with Your Documents (Next-Level AI)")

# Session memory
if "messages" not in st.session_state:
    st.session_state.messages = []

# Upload multiple files
files = st.file_uploader(
    "Upload PDF, DOCX, TXT",
    type=["pdf", "docx", "txt"],
    accept_multiple_files=True
)

all_text = ""

if files:
    for file in files:
        name = file.name.lower()

        if name.endswith(".pdf"):
            all_text += extract_text_from_pdf(file)

        elif name.endswith(".docx"):
            all_text += extract_text_from_docx(file)

        elif name.endswith(".txt"):
            all_text += extract_text_from_txt(file)

    st.success("✅ Documents loaded!")

    chunks = split_text(all_text)

    index, embeddings = create_vector_store(chunks)

# ------------------------------
# 💬 Chat UI
# ------------------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# Input box
user_input = st.chat_input("Ask something about your document...")

if user_input and files:

    # Show user message
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.write(user_input)

    # Search relevant chunks
    relevant_chunks = search_chunks(user_input, chunks, index, embeddings)
    context = " ".join(relevant_chunks)

    # AI response
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        full_response = ""

        stream = get_answer_stream(context, user_input)

        for chunk in stream:
            full_response += chunk
            response_placeholder.markdown(full_response)

    # Save response
    st.session_state.messages.append({"role": "assistant", "content": full_response})

    # Show source
    with st.expander("📄 Source Context"):
        st.write(context)