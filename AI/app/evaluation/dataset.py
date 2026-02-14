"""
Evaluation dataset for RAG testing.
Each sample contains:
- question
- expected document (for retriever evaluation)
- ground truth answer (for full RAG evaluation)
"""

evaluation_set = [
    {
        "question": "How long do refunds take?",
        "expected_doc": "Refunds usually take 5-10 business days.",
        "ground_truth_answer": "Refunds usually take 5-10 business days.",
    },
    {
        "question": "Are food items refundable?",
        "expected_doc": "Food items are non-refundable.",
        "ground_truth_answer": "Food items are non-refundable.",
    },
    {
        "question": "When are transactions flagged?",
        "expected_doc": "Transactions above 500 EUR are flagged for review.",
        "ground_truth_answer": "Transactions above 500 EUR are flagged for review.",
    },
    {
        "question": "What are possible fraud indicators?",
        "expected_doc": "Potential fraud indicators include high transaction frequency, unusual transaction amounts, and cross-border transactions.",
        "ground_truth_answer": "Potential fraud indicators include high transaction frequency, unusual transaction amounts, and cross-border transactions.",
    },
]
