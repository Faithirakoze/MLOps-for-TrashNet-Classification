import json
import numpy as np
import tensorflow as tf

from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

from src.retraining import (
    prepare_combined_training_dataset,
    split_retraining_dataset
)


# --------------------------------------------------
# 1. Define paths
# --------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent

ORIGINAL_MODEL_PATH = PROJECT_DIR / "models" / "trashnet_best_model.h5"
CANDIDATE_MODEL_PATH = PROJECT_DIR / "models" / "trashnet_retrained_candidate.keras"
CLASS_NAMES_PATH = PROJECT_DIR / "models" / "class_names.json"


# --------------------------------------------------
# 2. Load class names
# --------------------------------------------------

with open(CLASS_NAMES_PATH, "r") as file:
    class_names = json.load(file)


# --------------------------------------------------
# 3. Recreate the validation dataset
# --------------------------------------------------

print("\nPreparing validation dataset...")

X, y, summary = prepare_combined_training_dataset()

X_train, X_val, y_train, y_val = split_retraining_dataset(
    X,
    y,
    validation_size=0.2
)

print(f"Validation images: {len(X_val)}")
print(f"Validation image shape: {X_val.shape}")


# --------------------------------------------------
# 4. Load both models
# --------------------------------------------------

print("\nLoading models...")

original_model = tf.keras.models.load_model(
    ORIGINAL_MODEL_PATH,
    compile=False
)

candidate_model = tf.keras.models.load_model(
    CANDIDATE_MODEL_PATH,
    compile=False
)


# --------------------------------------------------
# 5. Define evaluation function
# --------------------------------------------------

def evaluate_model(model, model_name, X_val, y_val):
    """
    Evaluate a model using the validation dataset.

    Returns predictions and aggregate evaluation metrics.
    """

    probabilities = model.predict(X_val, verbose=0)

    predictions = np.argmax(probabilities, axis=1)

    metrics = {
        "accuracy": accuracy_score(y_val, predictions),
        "precision": precision_score(
            y_val, predictions, average="macro", zero_division=0
        ),
        "recall": recall_score(
            y_val, predictions, average="macro", zero_division=0
        ),
        "macro_f1": f1_score(
            y_val, predictions, average="macro", zero_division=0
        )
    }

    print(f"\n{'=' * 55}")
    print(f"{model_name} - VALIDATION RESULTS")
    print(f"{'=' * 55}")

    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")

    print("\nClassification Report:")
    print(
        classification_report(
            y_val,
            predictions,
            labels=list(range(len(class_names))),
            target_names=class_names,
            zero_division=0
        )
    )

    print("Confusion Matrix:")
    print(
        confusion_matrix(
            y_val,
            predictions,
            labels=list(range(len(class_names)))
        )
    )

    return predictions, metrics


# --------------------------------------------------
# 6. Evaluate both models
# --------------------------------------------------

original_predictions, original_metrics = evaluate_model(
    original_model,
    "Original Model",
    X_val,
    y_val
)

candidate_predictions, candidate_metrics = evaluate_model(
    candidate_model,
    "Retrained Candidate",
    X_val,
    y_val
)


# --------------------------------------------------
# 7. Compare the results
# --------------------------------------------------

print("\n" + "=" * 55)
print("MODEL COMPARISON ON THE SAME VALIDATION DATA")
print("=" * 55)

print(
    f"{'Metric':<15}"
    f"{'Original':<15}"
    f"{'Candidate':<15}"
    f"{'Difference':<15}"
)

for metric in original_metrics:
    original_value = original_metrics[metric]
    candidate_value = candidate_metrics[metric]

    difference = candidate_value - original_value

    print(
        f"{metric:<15}"
        f"{original_value:<15.4f}"
        f"{candidate_value:<15.4f}"
        f"{difference:+.4f}"
    )