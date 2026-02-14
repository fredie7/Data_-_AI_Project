"""
End-to-end RAG evaluation.
Measures answer accuracy.
"""

from typing import List, Dict


def evaluate_rag(chain, evaluation_set: List[Dict]):
    correct = 0

    for sample in evaluation_set:
        result = chain.invoke(sample["question"])
        answer = result["result"]

        if sample["ground_truth_answer"].lower() in answer.lower():
            correct += 1

    accuracy = correct / len(evaluation_set)

    return {
        "Answer Accuracy": round(accuracy, 3)
    }
