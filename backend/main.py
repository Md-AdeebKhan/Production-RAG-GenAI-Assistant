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
        chunk_size=800,
        chunk_overlap=150
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
            "k": 12,
            "fetch_k": 40
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

Rules:
- Use the context to find the answer.
- If the question uses different wording but refers to the same concept in the document, still answer using the context.
- Consider related words, synonyms, or paraphrased questions when matching the answer.
- Do NOT invent information that is not present in the document.
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

        # Combine retrieved chunks
        context = "\n\n".join(doc.page_content for doc in docs)

        # Ensure first chunk is included (often contains headers / names)
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