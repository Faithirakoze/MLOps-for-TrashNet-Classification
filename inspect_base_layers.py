from tensorflow.keras.models import load_model

MODEL_PATH = "models/trashnet_best_model.h5"

model = load_model(MODEL_PATH)

# Get the MobileNetV2 base model.
base_model = model.layers[2]

print("\nMOBILENETV2 LAYER INFORMATION")
print("--------------------------------")

print("Total layers:", len(base_model.layers))

print("\nLast 30 layers:")

for index, layer in enumerate(base_model.layers[-30:]):
    print(
        index + len(base_model.layers) - 30,
        layer.name,
        layer.trainable
    )
