
import json
import math
import os

from pathlib import Path


# -----------------------------
# PROJECT DIRECTORIES
# -----------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODELS_DIR = PROJECT_ROOT / "models"
REGISTRY_DIR = MODELS_DIR / "registry"
REGISTRY_FILE = REGISTRY_DIR / "model_registry.json"


# -----------------------------
# LOAD REGISTRY
# -----------------------------

def load_registry():
    """Load the model registry from its JSON file."""

    if not REGISTRY_FILE.exists():
        raise FileNotFoundError(
            f"Model registry not found: {REGISTRY_FILE}"
        )

    with open(REGISTRY_FILE, "r", encoding="utf-8") as file:
        registry = json.load(file)

    if "production_version" not in registry:
        raise ValueError("Registry is missing production_version.")

    if "models" not in registry:
        raise ValueError("Registry is missing models.")

    if registry["production_version"] not in registry["models"]:
        raise ValueError(
            "Production version does not exist in the registry."
        )

    return registry


# -----------------------------
# SAVE REGISTRY
# -----------------------------

def save_registry(registry):
    """
    Save registry updates safely using a temporary file.

    This reduces the risk of corrupting the registry
    if the process is interrupted during writing.
    """

    REGISTRY_DIR.mkdir(parents=True, exist_ok=True)

    temporary_file = REGISTRY_FILE.with_suffix(".tmp")

    with open(temporary_file, "w", encoding="utf-8") as file:
        json.dump(registry, file, indent=4)

        file.flush()
        os.fsync(file.fileno())

    os.replace(temporary_file, REGISTRY_FILE)


# -----------------------------
# GET PRODUCTION MODEL
# -----------------------------

def get_production_model():
    """Retrieve metadata for the current production model."""

    registry = load_registry()

    production_version = registry["production_version"]

    production = registry["models"][production_version]

    if production["status"] != "production":
        raise ValueError(
            "Registry production_version does not have production status."
        )

    return {
        "version": production_version,
        **production
    }


# -----------------------------
# REGISTER NEW MODEL
# -----------------------------

def register_model(
    version,
    filename,
    metrics,
    status="candidate"
):
    """Register a new model version."""

    registry = load_registry()

    if version in registry["models"]:
        raise ValueError(
            f"Model version {version} already exists."
        )

    if status != "candidate":
        raise ValueError(
            "New models must initially be registered as candidates."
        )

    model_path = MODELS_DIR / filename

    if not model_path.is_file():
        raise FileNotFoundError(
            f"Model file not found: {model_path}"
        )

    if not isinstance(metrics, dict):
        raise ValueError("Model metrics must be a dictionary.")

    registry["models"][version] = {
        "file": filename,
        "status": "candidate",
        "metrics": metrics
    }

    save_registry(registry)

    return registry["models"][version]


# -----------------------------
# UPDATE MODEL STATUS
# -----------------------------

def update_model_status(
    version,
    status,
    reason=None
):
    """Update the status of a non-production model."""

    allowed_statuses = {
        "candidate",
        "archived",
        "rejected"
    }

    if status not in allowed_statuses:
        raise ValueError(
            f"Invalid status transition: {status}. "
            "Use promote_model() to promote a model."
        )

    registry = load_registry()

    if version not in registry["models"]:
        raise ValueError(
            f"Model version {version} is not registered."
        )

    if registry["models"][version]["status"] == "production":
        raise ValueError(
            "Cannot change the production model status directly."
        )

    registry["models"][version]["status"] = status

    if reason:
        registry["models"][version]["reason"] = reason
    else:
        registry["models"][version].pop("reason", None)

    save_registry(registry)

    return registry["models"][version]


# -----------------------------
# PROMOTE CANDIDATE MODEL
# -----------------------------

def promote_model(version, promotion_decision):
    """
    Promote a candidate only when evaluation approves it.

    The previous production model is archived.
    Its model artifact is preserved.
    """

    if not isinstance(promotion_decision, dict):
        raise ValueError(
            "Promotion decision must be a dictionary."
        )

    if promotion_decision.get("approved") is not True:
        raise ValueError(
            "Promotion denied: candidate did not pass evaluation."
        )

    registry = load_registry()

    if version not in registry["models"]:
        raise ValueError(
            f"Model version {version} is not registered."
        )

    candidate = registry["models"][version]

    if candidate["status"] != "candidate":
        raise ValueError(
            f"Cannot promote model with status '{candidate['status']}'."
        )

    # Verify the candidate artifact.
    model_path = MODELS_DIR / candidate["file"]

    if not model_path.is_file():
        raise FileNotFoundError(
            f"Model file not found: {model_path}"
        )

    # Verify the current production model.
    previous_version = registry["production_version"]

    if previous_version not in registry["models"]:
        raise ValueError(
            "Current production model is missing from the registry."
        )

    previous_model = registry["models"][previous_version]

    if previous_model["status"] != "production":
        raise ValueError(
            "Current production model has an invalid status."
        )

    # Update both statuses in memory before saving.
    previous_model["status"] = "archived"

    candidate["status"] = "production"
    candidate["reason"] = promotion_decision.get(
        "reason",
        "Candidate passed promotion criteria."
    )

    registry["production_version"] = version

    # Save the complete registry update.
    save_registry(registry)

    return {
        "previous_production": previous_version,
        "new_production": version,
        "file": candidate["file"]
    }
