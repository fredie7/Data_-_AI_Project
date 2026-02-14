"""
Retriever evaluation metrics:
- Hit@k
- Mean Reciprocal Rank (MRR)
"""

from typing import List, Dict


def evaluate_retriever(vectorstore, evaluation_set: List[Dict], k: int = 3):
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})

    hits = 0
    reciprocal_ranks = []

    for sample in evaluation_set:
        docs = retriever.get_relevant_documents(sample["question"])
        retrieved_texts = [doc.page_content for doc in docs]

        # Hit@k
        if sample["expected_doc"] in retrieved_texts:
            hits += 1
            rank = retrieved_texts.index(sample["expected_doc"]) + 1
            reciprocal_ranks.append(1 / rank)
        else:
            reciprocal_ranks.append(0)

    hit_at_k = hits / len(evaluation_set)
    mrr = sum(reciprocal_ranks) / len(evaluation_set)

    return {
        "Hit@{}".format(k): round(hit_at_k, 3),
        "MRR": round(mrr, 3),
    }
