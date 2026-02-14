"""
Creates a RetrievalQA chain for policy lookup.
"""

from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from core.config import OPENAI_API_KEY
from rag.vectorstore import build_vectorstore


def build_policy_retriever():
    # LLM used for answering policy questions
    llm = ChatOpenAI(
        model="gpt-4o",
        temperature=0.0,
        openai_api_key=OPENAI_API_KEY
    )

    # Vector search backend
    vectorstore = build_vectorstore()

    # Configure retriever explicitly
    retriever = vectorstore.as_retriever(
        search_type="similarity",  # or "mmr"
        search_kwargs={"k": 3}
    )

    # Retrieval-based QA chain
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever
    )
