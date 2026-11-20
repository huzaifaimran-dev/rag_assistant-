import streamlit as st
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from groq import Groq
import os
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# Configure Groq API
API_KEY = os.environ.get("GROQ_API_KEY", "")
client = Groq(api_key=API_KEY)

# Function to read all uploaded PDFs
def load_pdfs(uploaded_files):
    text = ""
    for pdf in uploaded_files:
        reader = PdfReader(pdf)
        for page in reader.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted
    return text


# Function to split text into chunks
def get_chunks(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=200
    )
    return splitter.split_text(text)


# Function to find relevant chunks (word-overlap scoring)
def find_relevant_chunks(question, chunks, top_k=5):
    question_words = set(question.lower().split())
    scored_chunks = []
    for chunk in chunks:
        chunk_words = set(chunk.lower().split())
        score = len(question_words.intersection(chunk_words))
        scored_chunks.append((score, chunk))
    scored_chunks.sort(reverse=True, key=lambda x: x[0])
    return [chunk for score, chunk in scored_chunks[:top_k]]


# Function to get answer from Groq
def get_answer(question, relevant_chunks):
    context = "\n\n".join(relevant_chunks)
    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[
            {
                "role": "system",
                "content": """You are a helpful assistant that answers questions using only the
                document content provided. Give clear, well-structured answers.
                If the answer is not in the provided documents, say
                'This isn't covered in the uploaded documents.'"""
            },
            {
                "role": "user",
                "content": f"Document Content:\n{context}\n\nQuestion: {question}"
            }
        ]
    )
    return response.choices[0].message.content


# Streamlit app
st.set_page_config(page_title="PDF Q&A", page_icon="📄")
st.title("📄 PDF Q&A Assistant")
st.write("Upload any PDF(s) and ask questions about their content.")

# File uploader
uploaded_files = st.file_uploader(
    "Upload PDF(s)", type="pdf", accept_multiple_files=True
)

if uploaded_files:
    # Re-process if the set of uploaded files changes
    file_signature = tuple(f.name for f in uploaded_files)
    if st.session_state.get("file_signature") != file_signature:
        with st.spinner("📚 Reading and indexing PDFs... Please wait..."):
            raw_text = load_pdfs(uploaded_files)
            st.session_state.chunks = get_chunks(raw_text)
            st.session_state.file_signature = file_signature
            st.session_state.chat_history = []
        st.success(f"✅ Documents loaded! {len(st.session_state.chunks)} sections ready!")

    # Chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Show previous messages
    for message in st.session_state.chat_history:
        st.chat_message(message["role"]).write(message["content"])

    # User input
    question = st.chat_input("Ask a question about your PDF(s)...")

    if question:
        st.chat_message("user").write(question)
        st.session_state.chat_history.append({"role": "user", "content": question})

        with st.spinner("🔍 Searching documents..."):
            relevant_chunks = find_relevant_chunks(question, st.session_state.chunks)
            answer = get_answer(question, relevant_chunks)

        st.chat_message("assistant").write(answer)
        st.session_state.chat_history.append({"role": "assistant", "content": answer})
else:
    st.info("👆 Upload one or more PDFs to get started.")