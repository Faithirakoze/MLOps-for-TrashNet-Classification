
import math


def evaluate_promotion(
    current_metrics,
    candidate_metrics,
    minimum_accuracy=0.80
):
    """
    Decide whether a candidate model qualifies for promotion.

    Promotion requirements:
    1. Candidate macro-F1 must improve over production.
    2. Candidate accuracy must meet the minimum threshold.

    Args:
        current_metrics (dict): Production model metrics.
        candidate_metrics (dict): Candidate model metrics.
        minimum_accuracy (float): Minimum acceptable accuracy.

    Returns:
        dict: Promotion decision, metrics comparison, and explanation.
    """

    required_metrics = ["accuracy", "macro_f1"]

    # Validate metric dictionaries.
    for model_name, metrics in [
        ("Production", current_metrics),
        ("Candidate", candidate_metrics)
    ]:

        if not isinstance(metrics, dict):
            raise ValueError(
                f"{model_name} metrics must be a dictionary."
            )

        for metric in required_metrics:

            if metric not in metrics:
                raise ValueError(
                    f"{model_name} metrics are missing: {metric}"
                )

            value = metrics[metric]

            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or not 0 <= value <= 1
            ):
                raise ValueError(
                    f"{model_name} {metric} must be a valid "
                    "number between 0 and 1."
                )

    if (
        isinstance(minimum_accuracy, bool)
        or not isinstance(minimum_accuracy, (int, float))
        or not math.isfinite(minimum_accuracy)
        or not 0 <= minimum_accuracy <= 1
    ):
        raise ValueError(
            "minimum_accuracy must be between 0 and 1."
        )

    # Extract metrics.
    current_accuracy = current_metrics["accuracy"]
    candidate_accuracy = candidate_metrics["accuracy"]

    current_f1 = current_metrics["macro_f1"]
    candidate_f1 = candidate_metrics["macro_f1"]

    # Calculate improvements.
    f1_improvement = candidate_f1 - current_f1
    accuracy_change = candidate_accuracy - current_accuracy

    # Evaluate promotion requirements.
    improves_f1 = candidate_f1 > current_f1
    meets_accuracy = candidate_accuracy >= minimum_accuracy

    approved = improves_f1 and meets_accuracy

    # Explain the decision.
    if approved:
        reason = "Candidate passed all promotion criteria."

    elif not improves_f1 and not meets_accuracy:
        reason = (
            "Candidate did not improve macro-F1 and "
            "failed the minimum accuracy requirement."
        )

    elif not improves_f1:
        reason = "Candidate did not improve macro-F1."

    else:
        reason = "Candidate did not meet the minimum accuracy."

    return {
        "approved": approved,
        "reason": reason,

        "current_accuracy": current_accuracy,
        "candidate_accuracy": candidate_accuracy,
        "accuracy_change": accuracy_change,

        "current_macro_f1": current_f1,
        "candidate_macro_f1": candidate_f1,
        "macro_f1_improvement": f1_improvement,

        "minimum_accuracy": minimum_accuracy
    }
