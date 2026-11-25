import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.utils.class_weight import compute_class_weight

IMAGE_SIZE = (224, 224)

def load_class_names(path):
    """Load the ordered class names used by the model output indices."""
    with open(path, "r") as file:
        return json.load(file)

def decode_resize_scale(path, label):
    """Decode RGB image data, resize to 224x224, and normalize pixels to [0, 1]."""
    image = tf.io.decode_image(tf.io.read_file(path), channels=3, expand_animations=False)
    image.set_shape([None, None, 3])
    image = tf.image.resize(image, IMAGE_SIZE)
    return tf.cast(image, tf.float32) / 255.0, label

def calculate_class_weights(integer_labels, class_count):
    """Calculate balanced class weights from integer labels in the training partition."""
    values = compute_class_weight("balanced", classes=np.arange(class_count),
                                  y=np.asarray(integer_labels))
    return {i: float(value) for i, value in enumerate(values)}

def create_dataset(dataframe, class_to_index, training=False, batch_size=32):
    """Build a TensorFlow dataset and apply augmentation only when training is True."""
    augmentation = keras.Sequential([
        keras.layers.RandomFlip("horizontal"),
        keras.layers.RandomRotation(0.08),
        keras.layers.RandomZoom(0.10)
    ])
    labels = dataframe.label.map(class_to_index).to_numpy(dtype=np.int32)
    dataset = tf.data.Dataset.from_tensor_slices((dataframe.filepath.to_numpy(), labels))
    if training:
        dataset = dataset.shuffle(len(dataframe), reshuffle_each_iteration=True)
    dataset = dataset.map(decode_resize_scale, num_parallel_calls=tf.data.AUTOTUNE)
    if training:
        dataset = dataset.map(lambda x,y: (augmentation(x, training=True), y),
                              num_parallel_calls=tf.data.AUTOTUNE)
    return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
