import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps


# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "trashnet_best_model.h5"
CLASS_NAMES_PATH = BASE_DIR / "models" / "class_names.json"

IMAGE_SIZE = (224, 224)


def load_class_names(path=CLASS_NAMES_PATH):
    """Load the ordered class names corresponding to model output indices."""
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def load_trained_model(path=MODEL_PATH):
    """Load the saved Keras waste classification model."""
    if not path.exists():
        raise FileNotFoundError(f"Model not found: {path}")

    return tf.keras.models.load_model(path)


def preprocess_image(image):
    """Convert an uploaded image to RGB, resize it, and scale pixels to [0, 1]."""
    image = ImageOps.exif_transpose(image)
    image = image.convert("RGB")
    image = image.resize(IMAGE_SIZE)

    image_array = np.asarray(image, dtype=np.float32) / 255.0

    return np.expand_dims(image_array, axis=0)


def predict_image(image, model, class_names):
    """Predict the waste category of one image and return class probabilities."""
    processed_image = preprocess_image(image)

    probabilities = model.predict(processed_image, verbose=0)[0]

    predicted_index = int(np.argmax(probabilities))

    probability_results = {
        class_names[i]: float(probabilities[i])
        for i in range(len(class_names))
    }

    sorted_probabilities = dict(
        sorted(
            probability_results.items(),
            key=lambda item: item[1],
            reverse=True
        )
    )

    return {
        "predicted_class": class_names[predicted_index],
        "confidence": float(probabilities[predicted_index]),
        "probabilities": sorted_probabilities
    }
