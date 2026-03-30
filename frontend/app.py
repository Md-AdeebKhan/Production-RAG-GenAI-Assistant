import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import tempfile

from backend.main import (
    load_and_split_pdf,
    create_vectorstore,
    create_rag_chain,
)

st.set_page_config(page_title="DeepSeek RAG", layout="wide")

st.title("📄 DeepSeek RAG - PDF Chat")
st.caption("Upload a PDF document and ask questions based on its content using Retrieval-Augmented Generation.")

# Store RAG chain in session
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None


# -------- CLEAR CHAT BUTTON --------
if st.button("Clear Chat"):
    st.session_state.rag_chain = None
    st.rerun()


# -------- FILE UPLOAD --------
uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

if uploaded_file is not None:

    # Save temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        temp_path = tmp_file.name

    with st.spinner("Processing PDF..."):
        docs = load_and_split_pdf(temp_path)
        vectorstore = create_vectorstore(docs)
        st.session_state.rag_chain = create_rag_chain(vectorstore)

    st.success("PDF processed successfully!")


# -------- QUESTION INPUT --------
if st.session_state.rag_chain:

    user_question = st.text_input("Ask a question about the document")

    if user_question:
        with st.spinner("Generating answer..."):
            response = st.session_state.rag_chain.invoke(user_question)

        st.markdown("### 📌 Answer")
        st.markdown(response)