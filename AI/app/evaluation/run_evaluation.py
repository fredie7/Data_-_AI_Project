"""
Runs full RAG evaluation pipeline.
"""

from rag.vectorstore import build_vectorstore
from rag.retriever import build_policy_retriever

from evaluation.dataset import evaluation_set
from evaluation.retriever_eval import evaluate_retriever
from evaluation.rag_eval import evaluate_rag


def run():
    print("Building vectorstore...")
    vectorstore = build_vectorstore()

    print("Building RAG chain...")
    rag_chain = build_policy_retriever()

    print("\nEvaluating Retriever...")
    retriever_metrics = evaluate_retriever(vectorstore, evaluation_set, k=3)

    print("Retriever Metrics:")
    for k, v in retriever_metrics.items():
        print(f"{k}: {v}")

    print("\nEvaluating Full RAG...")
    rag_metrics = evaluate_rag(rag_chain, evaluation_set)

    print("RAG Metrics:")
    for k, v in rag_metrics.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    run()
