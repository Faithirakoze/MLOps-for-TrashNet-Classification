import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers

def build_model(num_classes=6, dropout=0.4, l2_strength=1e-4):
    """Build MobileNetV2 transfer learning classifier for 224x224 RGB images scaled to [0, 1]."""
    backbone = keras.applications.MobileNetV2(
        input_shape=(224,224,3), include_top=False, weights="imagenet"
    )
    backbone.trainable = False
    inputs = keras.Input((224,224,3))
    x = layers.Rescaling(2.0, offset=-1.0)(inputs)
    x = backbone(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(num_classes, activation="softmax",
                           kernel_regularizer=regularizers.l2(l2_strength))(x)
    return keras.Model(inputs, outputs), backbone

def compile_model(model, learning_rate=1e-3):
    """Compile a multiclass image classifier using Adam and sparse categorical cross-entropy."""
    model.compile(optimizer=keras.optimizers.Adam(learning_rate),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model

def unfreeze_top_layers(backbone, count=30):
    """Unfreeze the final backbone layers while leaving BatchNormalization layers frozen."""
    backbone.trainable = True
    for layer in backbone.layers[:-count]:
        layer.trainable = False
    for layer in backbone.layers:
        if isinstance(layer, layers.BatchNormalization):
            layer.trainable = False
