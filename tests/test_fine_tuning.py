from src.retraining import configure_fine_tuning


def test_fine_tuning_configuration():
    """Test the fine-tuning configuration."""

    model = configure_fine_tuning()

    base_model = model.layers[2]

    trainable_layers = sum(
        1 for layer in base_model.layers if layer.trainable
    )

    frozen_layers = sum(
        1 for layer in base_model.layers if not layer.trainable
    )

    print("\nFINE-TUNING CONFIGURATION")
    print("-------------------------")

    print("Total MobileNetV2 layers:", len(base_model.layers))
    print("Trainable layers:", trainable_layers)
    print("Frozen layers:", frozen_layers)

    print("\nLast 20 layers:")

    for layer in base_model.layers[-20:]:
        print(layer.name, layer.trainable)

    print("\nModel compilation successful!")


if __name__ == "__main__":
    test_fine_tuning_configuration()
    