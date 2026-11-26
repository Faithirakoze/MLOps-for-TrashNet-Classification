from src.retraining import prepare_retraining_dataset


def test_prepare_retraining_dataset():
    """Test image preparation for model retraining."""

    X, y = prepare_retraining_dataset()

    print("\nRETRAINING DATASET PREPARATION")

    print("Image array shape:", X.shape)
    print("Label array shape:", y.shape)

    print("Image data type:", X.dtype)
    print("Label data type:", y.dtype)

    if len(X) > 0:
        print("Minimum pixel value:", X.min())
        print("Maximum pixel value:", X.max())

        print("\nNumerical labels:", y)

    else:
        print("No valid images found.")


if __name__ == "__main__":
    test_prepare_retraining_dataset()
    