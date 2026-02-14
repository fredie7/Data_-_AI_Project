"""
Unstructured policy documents used by the RAG pipeline.
These are embedded and searched using vector similarity.
"""

from langchain_core.documents import Document

policy_docs = [
    Document(
        page_content="Refunds usually take 5-10 business days.",
        metadata={"source": "policy"}
    ),
    Document(
        page_content="Food items are non-refundable.",
        metadata={"source": "policy"}
    ),
    Document(
        page_content="Transactions above 500 EUR are flagged for review.",
        metadata={"source": "policy"}
    ),
    Document(
        page_content=(
            "Potential fraud indicators include high transaction frequency, "
            "unusual transaction amounts, and cross-border transactions."
        ),
        metadata={"source": "policy"}
    ),
]
