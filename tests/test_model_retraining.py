from src.retraining import (
    prepare_combined_training_dataset,
    split_retraining_dataset,
    retrain_model
)


def test_model_retraining():
    """Test the complete model retraining pipeline."""

    # Load the combined dataset.
    X, y, summary = prepare_combined_training_dataset()

    # Split the dataset.
    X_train, X_val, y_train, y_val = split_retraining_dataset(X, y)

    print("\nDATASET READY")
    print("Training images:", len(X_train))
    print("Validation images:", len(X_val))

    # Start model retraining.
    history, best_macro_f1 = retrain_model(
        X_train,
        y_train,
        X_val,
        y_val
    )

    print("\nFINAL RESULTS")
    print("Best validation macro-F1:", best_macro_f1)


if __name__ == "__main__":
    test_model_retraining()
