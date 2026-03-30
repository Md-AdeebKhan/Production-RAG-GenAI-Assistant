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
        chunk_size=1200,
        chunk_overlap=250
    )

    return splitter.split_documents(docs)


# -------- VECTOR STORE --------
def create_vectorstore(documents):

    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5",
        encode_kwargs={"normalize_embeddings": True}
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
            "k": 20,
            "fetch_k": 60
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
You are an intelligent document assistant.

Answer the user's question using the provided document context.

Instructions:
- Use the context to determine the best possible answer.
- The question may use different wording, synonyms, or paraphrasing.
- If the question is related to the information in the document, answer using the relevant context.
- Do NOT invent information that does not exist in the context.
- If the answer truly does not exist in the provided context, respond with: "I don't know."

Context:
{context}

Question:
{question}

Answer:
"""
    )

    def format_docs(docs):

        if not docs:
            return ""

        # combine retrieved chunks
        context = "\n\n".join(doc.page_content for doc in docs)

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