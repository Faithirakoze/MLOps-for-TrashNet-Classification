from src.retraining import prepare_combined_training_dataset


def test_combined_dataset():
    """Test combining original and uploaded training images."""

    X, y, summary = prepare_combined_training_dataset()

    print("\nCOMBINED TRAINING DATASET")

    for category, statistics in summary.items():

        print(f"\n{category}:")
        print(f"Original images: {statistics['original']}")
        print(f"Uploaded images: {statistics['uploaded']}")
        print(f"Duplicates: {statistics['duplicates']}")
        print(f"Invalid images: {statistics['invalid']}")

    print("\nTOTAL DATASET")
    print("Image shape:", X.shape)
    print("Label shape:", y.shape)

    print("\nTotal images:", len(X))


if __name__ == "__main__":
    test_combined_dataset()
