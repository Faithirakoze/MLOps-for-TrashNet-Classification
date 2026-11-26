from pathlib import Path

from src.retraining import (
    validate_image,
    save_uploaded_image,
    UPLOAD_DIR
)


def test_image_upload():
    """Test image validation and storage using an existing dataset image."""

    # Replace this with the path to an actual image on your computer.
    image_path = Path("data/test/plastic/plastic5.jpg")

    # Read the image as bytes.
    with open(image_path, "rb") as image_file:
        image_contents = image_file.read()

    # Validate the image.
    is_valid = validate_image(image_contents)
    print("Image validation:", is_valid)

    # Save the validated image.
    filename = save_uploaded_image(image_contents, "plastic")

    print("Image saved successfully!")
    print("Filename:", filename)
    print("Saved location:", UPLOAD_DIR / "plastic" / filename)


if __name__ == "__main__":
    test_image_upload()
