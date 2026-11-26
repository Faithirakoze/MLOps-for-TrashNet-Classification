
from src.model_promotion import evaluate_promotion


# Original model validation metrics
original_metrics = {
    "accuracy": 0.9457,
    "macro_f1": 0.9372
}

# Retrained candidate validation metrics
candidate_metrics = {
    "accuracy": 0.9284,
    "macro_f1": 0.9213
}


# Evaluate the candidate
result = evaluate_promotion(
    original_metrics,
    candidate_metrics
)


print("\nMODEL PROMOTION DECISION")
print("------------------------")
print(f"Approved: {result['approved']}")
print(f"Reason: {result['reason']}")
print(f"Original macro-F1: {result['current_macro_f1']:.4f}")
print(f"Candidate macro-F1: {result['candidate_macro_f1']:.4f}")
