from src.retraining import inspect_retraining_dataset


def test_dataset_inspection():
    """Test inspection of uploaded retraining images."""

    results = inspect_retraining_dataset()

    print("\nRETRAINING DATASET SUMMARY")

    for category, statistics in results["categories"].items():
        print(f"\n{category}:")
        print(f"Valid images: {statistics['valid']}")
        print(f"Invalid images: {statistics['invalid']}")
        print(f"Duplicates: {statistics['duplicates']}")

    print("\nTOTALS")
    print(f"Valid: {results['total_valid']}")
    print(f"Invalid: {results['total_invalid']}")
    print(f"Duplicates: {results['total_duplicates']}")


if __name__ == "__main__":
    test_dataset_inspection()
    