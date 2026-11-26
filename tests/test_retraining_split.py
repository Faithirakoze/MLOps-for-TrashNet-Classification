from src.retraining import (
    prepare_combined_training_dataset,
    split_retraining_dataset
)


def test_retraining_split():
    """Test the training and validation split."""

    # Load the combined dataset.
    X, y, summary = prepare_combined_training_dataset()

    # Split into training and validation sets.
    X_train, X_val, y_train, y_val = split_retraining_dataset(X, y)

    print("\nRETRAINING DATASET SPLIT")
    print("------------------------")

    print("Training images:", len(X_train))
    print("Validation images:", len(X_val))

    print("Training labels:", len(y_train))
    print("Validation labels:", len(y_val))

    print("\nTraining image shape:", X_train.shape)
    print("Validation image shape:", X_val.shape)

    print("\nTraining class distribution:")

    for index, category in enumerate(
        ["cardboard", "glass", "metal", "paper", "plastic", "trash"]
    ):
        print(f"{category}: {(y_train == index).sum()}")

    print("\nValidation class distribution:")

    for index, category in enumerate(
        ["cardboard", "glass", "metal", "paper", "plastic", "trash"]
    ):
        print(f"{category}: {(y_val == index).sum()}")


if __name__ == "__main__":
    test_retraining_split()
    