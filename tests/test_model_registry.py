
from src.model_registry import (
    load_registry,
    get_production_model,
    promote_model,
    register_model
)


def main():
    # Display the current production model.
    production = get_production_model()

    print("\nCURRENT PRODUCTION MODEL")
    print("------------------------")
    print(f"Version: {production['version']}")
    print(f"File: {production['file']}")
    print(f"Status: {production['status']}")
    print(f"Accuracy: {production['metrics']['accuracy']}")
    print(f"Macro-F1: {production['metrics']['macro_f1']}")

    # Display all registered versions.
    registry = load_registry()

    print("\nREGISTERED MODEL VERSIONS")
    print("-------------------------")

    for version, details in registry["models"].items():
        print(f"{version}: {details['status']}")

    # Try promoting the rejected candidate.
    print("\nTESTING PROMOTION SAFEGUARD")
    print("---------------------------")

    rejected_decision = {
        "approved": False,
        "reason": "Candidate did not improve macro-F1."
    }

    try:
        promote_model("1.1.0", rejected_decision)
    except ValueError as error:
        print(error)

    # Test registering a new candidate.
    print("\nTESTING MODEL REGISTRATION")
    print("--------------------------")

    try:
        register_model(
            version="1.2.0",
            filename="trashnet_retrained_candidate.keras",
            metrics={
                "accuracy": 0.85,
                "macro_f1": 0.83
            }
        )

        print("New model version registered successfully.")

    except (ValueError, FileNotFoundError) as error:
        print(f"Registration failed: {error}")


if __name__ == "__main__":
    main()
