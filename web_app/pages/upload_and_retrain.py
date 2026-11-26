
import streamlit as st
import requests
import sys
import zipfile
import io
import hashlib
import os

from pathlib import Path
from PIL import Image, UnidentifiedImageError


# -----------------------------
# Project configuration
# -----------------------------

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Make the project root accessible for Python imports.
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.mlops_pipeline import run_mlops_pipeline


CLASSES = [
    "cardboard",
    "glass",
    "metal",
    "paper",
    "plastic",
    "trash"
]

ALLOWED_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".webp"
}


# -----------------------------
# Streamlit configuration
# -----------------------------

st.set_page_config(
    page_title="Upload and Retrain",
    layout="wide"
)

st.title("Upload and Retrain")

st.write(
    "Upload new labelled waste images and manage the "
    "model retraining workflow."
)

st.divider()


# -----------------------------
# Helper: Validate image
# -----------------------------

def validate_image(image_bytes):
    """Check whether uploaded bytes represent a readable image."""

    try:
        with Image.open(io.BytesIO(image_bytes)) as image:
            image.verify()

        return True

    except (UnidentifiedImageError, OSError, ValueError):
        return False


# -----------------------------
# Helper: Process ZIP file
# -----------------------------

def process_zip_file(zip_file):
    """
    Read images from a ZIP archive.

    Expected structure:

        cardboard/image1.jpg
        glass/image2.jpg
        metal/image3.jpg
        paper/image4.jpg
        plastic/image5.jpg
        trash/image6.jpg
    """

    categorized_images = {
        category: [] for category in CLASSES
    }

    rejected_files = []
    seen_hashes = set()

    try:
        with zipfile.ZipFile(io.BytesIO(zip_file.getvalue())) as archive:

            for zip_info in archive.infolist():

                if zip_info.is_dir():
                    continue

                file_path = Path(zip_info.filename)

                # Ignore hidden files and operating-system metadata.
                if any(part.startswith(".") for part in file_path.parts):
                    continue

                if "__MACOSX" in file_path.parts:
                    continue

                if file_path.suffix.lower() not in ALLOWED_EXTENSIONS:
                    continue

                if len(file_path.parts) < 2:
                    rejected_files.append(
                        (zip_info.filename, "Missing category folder")
                    )
                    continue

                # Find a recognized category folder.
                category = next(
                    (
                        part.lower()
                        for part in file_path.parts[:-1]
                        if part.lower() in CLASSES
                    ),
                    None
                )

                if category is None:
                    rejected_files.append(
                        (zip_info.filename, "No recognized category folder")
                    )
                    continue

                try:
                    image_bytes = archive.read(zip_info)

                    if not validate_image(image_bytes):
                        rejected_files.append(
                            (zip_info.filename, "Invalid or unreadable image")
                        )
                        continue

                    # Detect identical files within this ZIP.
                    image_hash = hashlib.sha256(
                        image_bytes
                    ).hexdigest()

                    if image_hash in seen_hashes:
                        rejected_files.append(
                            (zip_info.filename, "Duplicate image in ZIP")
                        )
                        continue

                    seen_hashes.add(image_hash)

                    categorized_images[category].append({
                        "name": file_path.name,
                        "bytes": image_bytes
                    })

                except Exception as error:
                    rejected_files.append(
                        (zip_info.filename, str(error))
                    )

    except zipfile.BadZipFile:
        raise ValueError(
            "The uploaded file is not a valid ZIP archive."
        )

    total_images = sum(
        len(images)
        for images in categorized_images.values()
    )

    if total_images == 0:
        raise ValueError(
            "No valid labelled images were found in the ZIP file."
        )

    return categorized_images, rejected_files


# -----------------------------
# Helper: Send images to API
# -----------------------------

def upload_category_images(category, images):
    """Upload images for one category, one request per image."""

    uploaded_files = []
    rejected_files = []

    for image in images:
        response = requests.post(
            f"{API_URL}/retraining/upload",
            data={"category": category},
            files={
                "file": (
                    image["name"],
                    image["bytes"],
                    "application/octet-stream"
                )
            },
            timeout=60
        )

        if response.status_code == 200:
            uploaded_files.append(response.json())

        elif response.status_code in (400, 413):
            rejected_files.append({
                "filename": image["name"],
                "reason": response.json().get("detail", response.text)
            })

        else:
            raise RuntimeError(
                f"{category}: {response.status_code} - {response.text}"
            )

    return {
        "category": category,
        "uploaded_count": len(uploaded_files),
        "rejected_count": len(rejected_files),
        "uploaded_files": uploaded_files,
        "rejected_files": rejected_files
    }


# -----------------------------
# 1. Upload images
# -----------------------------

st.subheader("1. Upload New Training Images")

st.info(
    "Uploaded images are sent to the retraining API. "
    "They are not yet automatically included in the current "
    "MLOps training pipeline."
)

upload_method = st.radio(
    "Choose upload method",
    [
        "Upload by category",
        "Upload all categories (ZIP)"
    ],
    horizontal=True
)


# -----------------------------
# OPTION A: Upload by category
# -----------------------------

if upload_method == "Upload by category":

    st.info(
        "Select the correct category and upload images "
        "belonging to that category."
    )

    category = st.selectbox(
        "Select waste category",
        CLASSES
    )

    uploaded_files = st.file_uploader(
        "Choose images",
        type=["jpg", "jpeg", "png", "webp"],
        accept_multiple_files=True,
        key="category_uploader"
    )

    if uploaded_files:

        st.write(
            f"Selected images: **{len(uploaded_files)}**"
        )

        preview_columns = st.columns(
            min(len(uploaded_files), 4)
        )

        for index, uploaded_file in enumerate(uploaded_files):

            with preview_columns[index % len(preview_columns)]:

                st.image(
                    uploaded_file,
                    caption=uploaded_file.name,
                    use_container_width=True
                )

        if st.button(
            "Upload Images",
            type="primary"
        ):

            images = []

            for uploaded_file in uploaded_files:

                image_bytes = uploaded_file.getvalue()

                if not validate_image(image_bytes):
                    st.error(
                        f"{uploaded_file.name} is not a valid image."
                    )
                    continue

                images.append({
                    "name": uploaded_file.name,
                    "bytes": image_bytes
                })

            if not images:
                st.error("No valid images to upload.")

            else:
                try:
                    with st.spinner("Uploading images..."):

                        result = upload_category_images(
                            category,
                            images
                        )

                    st.session_state["upload_result"] = {
                        "method": "category",
                        "results": {
                            category: result
                        }
                    }

                    st.success("Upload completed.")

                except Exception as error:
                    st.error(f"Upload failed: {error}")


# -----------------------------
# OPTION B: Upload all categories
# -----------------------------

else:

    st.info(
        "Upload a ZIP file containing folders named "
        "cardboard, glass, metal, paper, plastic and trash."
    )

    zip_file = st.file_uploader(
        "Choose ZIP file",
        type=["zip"],
        key="zip_uploader"
    )

    if zip_file:

        try:
            categorized_images, rejected_files = process_zip_file(
                zip_file
            )

            total_images = sum(
                len(images)
                for images in categorized_images.values()
            )

            st.success(
                f"Found {total_images} valid images."
            )

            st.subheader("ZIP Upload Summary")

            summary_columns = st.columns(3)

            for index, category in enumerate(CLASSES):

                with summary_columns[index % 3]:

                    st.metric(
                        category.capitalize(),
                        len(categorized_images[category])
                    )

            if rejected_files:

                st.warning(
                    f"{len(rejected_files)} files were rejected."
                )

                with st.expander("View rejected files"):

                    for filename, reason in rejected_files:
                        st.write(
                            f"Rejected: {filename} — {reason}"
                        )

            if st.button(
                "Upload All Images",
                type="primary"
            ):

                upload_results = {}

                try:
                    progress = st.progress(0)

                    with st.spinner(
                        "Uploading all categories..."
                    ):

                        for index, category in enumerate(CLASSES):

                            images = categorized_images[category]

                            if images:

                                result = upload_category_images(
                                    category,
                                    images
                                )

                                upload_results[category] = result

                            progress.progress(
                                (index + 1) / len(CLASSES)
                            )

                    st.session_state["upload_result"] = {
                        "method": "zip",
                        "results": upload_results,
                        "rejected_files": rejected_files
                    }

                    st.success(
                        "ZIP upload process completed."
                    )

                except Exception as error:

                    st.error(
                        f"Upload process failed: {error}"
                    )

                    st.warning(
                        "Some categories may already have been uploaded. "
                        "Check the API upload storage before retrying."
                    )

        except Exception as error:

            st.error(
                f"Could not process ZIP file: {error}"
            )


# -----------------------------
# 2. Display upload results
# -----------------------------

if "upload_result" in st.session_state:

    stored_result = st.session_state["upload_result"]

    st.divider()
    st.subheader("2. Upload Results")

    total_uploaded = 0
    total_rejected = 0

    for category, result in stored_result["results"].items():

        total_uploaded += result.get("uploaded_count", 0)
        total_rejected += result.get("rejected_count", 0)

    col1, col2 = st.columns(2)

    col1.metric("Uploaded", total_uploaded)
    col2.metric("Rejected by API", total_rejected)

    st.write("**Results by category**")

    for category, result in stored_result["results"].items():

        with st.expander(category.capitalize()):

            st.write(
                f"Uploaded: {result.get('uploaded_count', 0)}"
            )

            st.write(
                f"Rejected: {result.get('rejected_count', 0)}"
            )

            for filename in result.get("uploaded_files", []):
                st.write(f"Uploaded: {filename}")

            for filename in result.get("rejected_files", []):
                st.write(f"Rejected: {filename}")

    if stored_result.get("rejected_files"):

        with st.expander("Files rejected during ZIP processing"):

            for filename, reason in stored_result["rejected_files"]:
                st.write(f"Rejected: {filename} — {reason}")


# -----------------------------
# 3. Run MLOps retraining pipeline
# -----------------------------

st.divider()

st.subheader("3. Run MLOps Retraining Pipeline")

st.write(
    "The pipeline prepares the existing training dataset, "
    "fine-tunes the production model, evaluates the candidate "
    "against the independent holdout dataset, and determines "
    "whether the candidate qualifies for promotion."
)

st.warning(
    "Retraining can take several minutes. "
    "Do not close this page while the pipeline is running."
)

if st.button(
    "Start Retraining Pipeline",
    type="primary"
):

    try:

        with st.spinner(
            "Running the MLOps pipeline. This may take a while..."
        ):

            results = run_mlops_pipeline()

        st.success(
            "MLOps pipeline completed successfully."
        )

        st.divider()
        st.subheader("Pipeline Results")

        col1, col2 = st.columns(2)

        with col1:

            st.metric(
                "Production Version",
                results["production_version"]
            )

            st.write("Production Metrics")

            st.json(
                results["current_metrics"]
            )

        with col2:

            st.metric(
                "Candidate Version",
                results["candidate_version"]
            )

            st.write("Candidate Metrics")

            st.json(
                results["candidate_metrics"]
            )

        st.subheader("Promotion Decision")

        st.write(
            results["promotion_decision"]
        )

    except Exception as error:

        st.error(
            f"MLOps pipeline failed: {error}"
        )

        st.exception(error)
