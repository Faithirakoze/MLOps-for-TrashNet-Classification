from tensorflow.keras.models import load_model

MODEL_PATH = "models/trashnet_best_model.h5"

model = load_model(MODEL_PATH)

print("\nMODEL SUMMARY")
model.summary()

print("\nMODEL LAYERS")
for index, layer in enumerate(model.layers):
    print(index, layer.name, layer.trainable, layer.output_shape if hasattr(layer, "output_shape") else "")
