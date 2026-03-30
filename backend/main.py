import os
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")

if not DEEPSEEK_API_KEY:
    raise ValueError("DEEPSEEK_API_KEY not found in environment variables")


# -------- LOAD AND SPLIT PDF --------
def load_and_split_pdf(file_path: str):

    loader = PyPDFLoader(file_path)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50
    )

    return splitter.split_documents(docs)


# -------- VECTOR STORE --------
def create_vectorstore(documents):

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings
    )

    return vectorstore


# -------- RAG CHAIN --------
def create_rag_chain(vectorstore):

    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 8,
            "fetch_k": 20
        }
    )

    llm = ChatOpenAI(
        api_key=DEEPSEEK_API_KEY,
        base_url="https://api.deepseek.com/v1",
        model="deepseek-chat",
        temperature=0
    )

    prompt = ChatPromptTemplate.from_template(
        """
You are a precise AI assistant.

Answer ONLY using the provided context.
If the answer is not found in the context, say "I don't know."

Context:
{context}

Question:
{question}
"""
    )

    def format_docs(docs):
        context = "\n\n".join(doc.page_content for doc in docs)

    # always include first chunk (usually contains name/header)
        if docs:
            context = docs[0].page_content + "\n\n" + context

        return context
    chain = (
        {
            "context": retriever | format_docs,
            "question": lambda x: x
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain