"""
Builds the FAISS vector store for policy documents.
"""

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from core.config import OPENAI_API_KEY
from rag.policies import policy_docs

def build_vectorstore():
    # Split documents into overlapping chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=50
    )
    chunks = splitter.split_documents(policy_docs)

    # Initialize OpenAI embeddings
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_key=OPENAI_API_KEY
    )

    # Build FAISS index
    return FAISS.from_documents(chunks, embeddings)
