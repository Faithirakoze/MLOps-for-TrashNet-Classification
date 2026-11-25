import time
from contextlib import asynccontextmanager
from io import BytesIO

from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError

from src.prediction import (
    load_class_names,
    load_trained_model,
    predict_image
)


# Application start time
START_TIME = time.monotonic()

# Maximum uploaded image size: 10 MB
MAX_FILE_SIZE = 10 * 1024 * 1024


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the trained model and class names when the API starts."""
    app.state.model = load_trained_model()
    app.state.class_names = load_class_names()

    if app.state.model.output_shape[-1] != len(app.state.class_names):
        raise RuntimeError("Model output size does not match class names.")

    yield


app = FastAPI(
    title="TrashNet Waste Classification API",
    description="API for classifying waste images using MobileNetV2.",
    version="1.0.0",
    lifespan=lifespan
)


@app.get("/health")
def health():
    """Check whether the API and trained model are ready."""
    model_loaded = (
        hasattr(app.state, "model")
        and hasattr(app.state, "class_names")
    )

    return {
        "status": "healthy" if model_loaded else "not_ready",
        "model_loaded": model_loaded
    }


@app.get("/uptime")
def uptime():
    """Return how long the API has been running in seconds."""
    elapsed = time.monotonic() - START_TIME

    return {
        "uptime_seconds": round(elapsed, 2)
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Accept one uploaded image and return its predicted waste category."""

    if file.content_type not in {
        "image/jpeg",
        "image/png",
        "image/webp"
    }:
        raise HTTPException(
            status_code=400,
            detail="Upload a JPEG, PNG, or WebP image."
        )

    contents = await file.read()

    if len(contents) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail="Image exceeds the 10 MB upload limit."
        )

    try:
        image = Image.open(BytesIO(contents))
        image.verify()

        image = Image.open(BytesIO(contents))

    except (UnidentifiedImageError, OSError):
        raise HTTPException(
            status_code=400,
            detail="The uploaded file is not a valid image."
        )

    result = predict_image(
        image=image,
        model=app.state.model,
        class_names=app.state.class_names
    )

    return result
