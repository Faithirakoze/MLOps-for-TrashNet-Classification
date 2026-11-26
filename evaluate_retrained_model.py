import json
import numpy as np

from pathlib import Path
from tensorflow.keras.models import load_model
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)

BASE_DIR = Path(__file__).resolve().parent

TEST_DIR = BASE_DIR / "data" / "test"
CLASS_NAMES_PATH = BASE_DIR / "models" / "class_names.json"

ORIGINAL_MODEL_PATH = (
    BASE_DIR / "models" / "trashnet_best_model.h5"
)

CANDIDATE_MODEL_PATH = (
    BASE_DIR / "models" / "trashnet_retrained_candidate.keras"
)


def load_test_dataset():
    """Load and preprocess the original test images."""

    images = []
    labels = []

    with open(CLASS_NAMES_PATH, "r") as file:
        class_names = json.load(file)

    class_indices = {
        name: index for index, name in enumerate(class_names)
    }

    from PIL import Image

    for category in class_names:

        category_dir = TEST_DIR / category

        for image_path in category_dir.iterdir():

            if not image_path.is_file():
                continue

            try:
                with Image.open(image_path) as image:

                    image = image.convert("RGB")
                    image = image.resize((224, 224))

                    image_array = np.array(
                        image,
                        dtype=np.float32
                    ) / 255.0

                    images.append(image_array)
                    labels.append(class_indices[category])

            except (OSError, ValueError):
                print(f"Skipping invalid image: {image_path}")

    return (
        np.array(images, dtype=np.float32),
        np.array(labels, dtype=np.int32),
        class_names
    )


def evaluate_model(model_path, X_test, y_test, class_names):
    """Evaluate a model using the same test dataset."""

    model = load_model(model_path)

    predictions = model.predict(X_test, verbose=0)
    predicted_labels = np.argmax(predictions, axis=1)

    results = {
        "accuracy": accuracy_score(y_test, predicted_labels),
        "precision": precision_score(
            y_test, predicted_labels,
            average="macro", zero_division=0
        ),
        "recall": recall_score(
            y_test, predicted_labels,
            average="macro", zero_division=0
        ),
        "macro_f1": f1_score(
            y_test, predicted_labels,
            average="macro", zero_division=0
        )
    }

    print("\nCLASSIFICATION REPORT")
    print(classification_report(
        y_test,
        predicted_labels,
        target_names=class_names,
        zero_division=0
    ))

    return results


if __name__ == "__main__":

    X_test, y_test, class_names = load_test_dataset()

    print("Test images:", len(X_test))

    print("\nEvaluating original model...")
    original_results = evaluate_model(
        ORIGINAL_MODEL_PATH,
        X_test,
        y_test,
        class_names
    )

    print("\nEvaluating retrained candidate...")
    candidate_results = evaluate_model(
        CANDIDATE_MODEL_PATH,
        X_test,
        y_test,
        class_names
    )

    print("\nMODEL COMPARISON")
    print("----------------")

    for metric in original_results:
        original = original_results[metric]
        candidate = candidate_results[metric]

        print(
            f"{metric}: "
            f"Original={original:.4f} | "
            f"Candidate={candidate:.4f} | "
            f"Difference={candidate - original:+.4f}"
        )
        