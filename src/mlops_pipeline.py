
import json
import re

import numpy as np
import tensorflow as tf

from PIL import Image

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

from src.retraining import (
    prepare_training_dataset,
    split_retraining_dataset,
    retrain_model,
    BASE_DIR,
    CLASS_NAMES
)

from src.model_promotion import evaluate_promotion

from src.model_registry import (
    get_production_model,
    register_model,
    promote_model,
    update_model_status,
    load_registry
)


MODELS_DIR = BASE_DIR / "models"
HOLDOUT_DIR = BASE_DIR / "data" / "holdout"
REGISTRY_PATH = MODELS_DIR / "registry" / "model_registry.json"



def get_next_candidate_version():
    """
    Generate the next unused patch version.

    Checks both the model registry and existing
    candidate model artifacts.
    """

    with open(REGISTRY_PATH, "r") as file:
        registry = json.load(file)

    used_versions = set(registry.get("models", {}).keys())

    # Include candidate files that are not registered yet.
    for path in MODELS_DIR.glob("trashnet_candidate_*.keras"):

        match = re.match(
            r"^(\d+)\.(\d+)\.(\d+)$",
            version
        )

        if match:
            used_versions.add(match.group(1))

    version_numbers = []

    for version in used_versions:

        match = re.match(
            r"^(\d+)\.(\d+)\.(\d+)$",
            version
        )

        if match:
            version_numbers.append(
                tuple(map(int, match.groups()))
            )

    if not version_numbers:
        return "1.0.0"

    major, minor, patch = max(version_numbers)

    next_version = f"{major}.{minor}.{patch + 1}"

    while next_version in used_versions:

        patch += 1
        next_version = f"{major}.{minor}.{patch + 1}"

    return next_version


def load_holdout_dataset():
    """
    Load the independent holdout dataset.

    This dataset is used exclusively for evaluating
    production and candidate models.
    """

    images = []
    labels = []
    class_counts = {}

    allowed_extensions = {
        ".jpg",
        ".jpeg",
        ".png",
        ".webp"
    }

    for label, category in enumerate(CLASS_NAMES):

        category_dir = HOLDOUT_DIR / category

        if not category_dir.exists():
            raise FileNotFoundError(
                f"Missing holdout folder: {category_dir}"
            )

        count = 0

        for image_path in sorted(category_dir.iterdir()):

            if not image_path.is_file():
                continue

            if image_path.suffix.lower() not in allowed_extensions:
                continue

            try:
                with Image.open(image_path) as image:

                    image = image.convert("RGB")
                    image = image.resize((224, 224))

                    image_array = np.asarray(
                        image,
                        dtype=np.float32
                    ) / 255.0

                images.append(image_array)
                labels.append(label)
                count += 1

            except (OSError, ValueError) as error:

                raise ValueError(
                    f"Could not read {image_path}: {error}"
                ) from error

        class_counts[category] = count

    missing_classes = [
        category
        for category, count in class_counts.items()
        if count == 0
    ]

    if missing_classes:
        raise ValueError(
            f"Holdout dataset is missing images for: {missing_classes}"
        )

    X_holdout = np.asarray(images, dtype=np.float32)
    y_holdout = np.asarray(labels, dtype=np.int32)

    print("\nINDEPENDENT HOLDOUT DATASET")
    print("---------------------------")

    for category, count in class_counts.items():
        print(f"{category}: {count}")

    print("Total images:", len(X_holdout))

    return X_holdout, y_holdout


def evaluate_model(model_path, X, y):
    """
    Evaluate a model on a supplied dataset.

    Returns accuracy, macro precision,
    macro recall and macro-F1.
    """

    model = tf.keras.models.load_model(
        model_path,
        compile=False
    )

    probabilities = model.predict(
        X,
        verbose=0
    )

    predictions = np.argmax(
        probabilities,
        axis=1
    )

    return {
        "accuracy": float(
            accuracy_score(y, predictions)
        ),

        "precision": float(
            precision_score(
                y,
                predictions,
                average="macro",
                zero_division=0
            )
        ),

        "recall": float(
            recall_score(
                y,
                predictions,
                average="macro",
                zero_division=0
            )
        ),

        "macro_f1": float(
            f1_score(
                y,
                predictions,
                average="macro",
                zero_division=0
            )
        )
    }


def run_mlops_pipeline(candidate_version=None):
    """
    Execute the complete TrashNet MLOps workflow.

    1. Load the current production model.
    2. Generate a unique candidate version.
    3. Prepare the original training dataset.
    4. Split training and validation data.
    5. Load the independent holdout dataset.
    6. Evaluate the current production model.
    7. Fine-tune the production model.
    8. Evaluate the candidate on the same holdout data.
    9. Apply promotion criteria.
    10. Register the candidate.
    11. Promote or reject the candidate.
    12. Report the final production model.
    """

    print("\n" + "=" * 55)
    print("TRASHNET MLOPS PIPELINE")
    print("=" * 55)


    print("\nSTEP 1: LOADING PRODUCTION MODEL")

    production = get_production_model()

    production_model_path = MODELS_DIR / production["file"]

    if not production_model_path.is_file():
        raise FileNotFoundError(production_model_path)

    print("Production version:", production["version"])
    print("Production file:", production_model_path.name)


    print("\nSTEP 2: GENERATING CANDIDATE VERSION")

    if candidate_version is None:
        candidate_version = get_next_candidate_version()

    if not re.fullmatch(
        r"\d+\.\d+\.\d+",
        candidate_version
    ):
        raise ValueError(
            f"Invalid candidate version: {candidate_version}"
        )

    registry = load_registry()

    if candidate_version in registry["models"]:
        raise ValueError(
            f"Version {candidate_version} is already registered."
        )

    candidate_model_path = (
        MODELS_DIR /
        f"trashnet_candidate_{candidate_version}.keras"
    )

    if candidate_model_path.exists():
        raise FileExistsError(
            f"Candidate artifact already exists: {candidate_model_path}"
        )

    print("Candidate version:", candidate_version)
    print("Candidate file:", candidate_model_path.name)


    print("\nSTEP 3: PREPARING TRAINING DATA")

    X, y, dataset_summary = prepare_training_dataset()

    if len(X) == 0:
        raise ValueError(
            "No valid training images were found."
        )

    class_counts = np.bincount(
        y,
        minlength=len(CLASS_NAMES)
    )

    for category, count in zip(CLASS_NAMES, class_counts):
        print(f"{category}: {count}")

    if np.any(class_counts < 2):
        raise ValueError(
            "Each class needs at least two images."
        )

    print("Total training images:", len(X))


    print("\nSTEP 4: SPLITTING TRAINING DATA")

    X_train, X_val, y_train, y_val = split_retraining_dataset(
        X,
        y,
        validation_size=0.2
    )

    print("Training images:", len(X_train))
    print("Validation images:", len(X_val))


    print("\nSTEP 5: LOADING INDEPENDENT HOLDOUT DATA")

    X_holdout, y_holdout = load_holdout_dataset()


    print("\nSTEP 6: EVALUATING PRODUCTION MODEL")

    current_metrics = evaluate_model(
        production_model_path,
        X_holdout,
        y_holdout
    )

    print("\nPRODUCTION MODEL METRICS")

    for metric, value in current_metrics.items():
        print(f"{metric}: {value:.4f}")


    print("\nSTEP 7: RETRAINING MODEL")

    history, best_validation_f1 = retrain_model(
        X_train,
        y_train,
        X_val,
        y_val,
        model_path=production_model_path,
        candidate_path=candidate_model_path
    )

    if not candidate_model_path.is_file():
        raise FileNotFoundError(
            "No candidate model was saved during retraining."
        )

    print(
        "Best validation macro-F1:",
        best_validation_f1
    )


    print("\nSTEP 8: EVALUATING CANDIDATE MODEL")

    candidate_metrics = evaluate_model(
        candidate_model_path,
        X_holdout,
        y_holdout
    )

    print("\nCANDIDATE MODEL METRICS")

    for metric, value in candidate_metrics.items():
        print(f"{metric}: {value:.4f}")

    print("\nSTEP 9: APPLYING PROMOTION CRITERIA")

    decision = evaluate_promotion(
        current_metrics,
        candidate_metrics,
        minimum_accuracy=0.80
    )

    print("Approved:", decision["approved"])
    print("Reason:", decision["reason"])


    print("\nSTEP 10: REGISTERING CANDIDATE")

    register_model(
        version=candidate_version,
        filename=candidate_model_path.name,
        metrics=candidate_metrics,
        status="candidate"
    )

    print("\nSTEP 11: UPDATING MODEL REGISTRY")

    if decision["approved"]:

        result = promote_model(
            candidate_version,
            decision
        )

        print("Candidate promoted.")
        print(
            "Previous production:",
            result["previous_production"]
        )
        print(
            "New production:",
            result["new_production"]
        )

    else:

        update_model_status(
            candidate_version,
            "rejected",
            reason=decision["reason"]
        )

        print("Candidate rejected.")
        print("Previous production remains unchanged.")


    final_production = get_production_model()

    print("\n" + "=" * 55)
    print("MLOPS PIPELINE COMPLETED")
    print("=" * 55)

    print(
        "Production version:",
        final_production["version"]
    )

    print(
        "Production model:",
        final_production["file"]
    )

    return {
        "production_version": final_production["version"],
        "current_metrics": current_metrics,
        "candidate_metrics": candidate_metrics,
        "promotion_decision": decision,
        "candidate_version": candidate_version
    }


if __name__ == "__main__":
    run_mlops_pipeline()
