from pathlib import Path
from io import BytesIO
import hashlib
import sqlite3
from datetime import datetime, timezone

import numpy as np
import tensorflow as tf
from PIL import Image, UnidentifiedImageError

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.utils.class_weight import compute_class_weight

from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (
    Callback,
    EarlyStopping,
    ReduceLROnPlateau
)

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "trashnet_best_model.h5"

TRAIN_DIR = BASE_DIR / "data" / "train"

UPLOAD_DIR = BASE_DIR / "data" / "retraining_uploads"

DATABASE_PATH = BASE_DIR / "data" / "retraining_uploads.db"

CANDIDATE_MODEL_PATH = (
    BASE_DIR / "models" / "trashnet_retrained_candidate.keras"
)

CLASS_NAMES = [
    "cardboard",
    "glass",
    "metal",
    "paper",
    "plastic",
    "trash"
]

IMAGE_SIZE = (224, 224)

MAX_FILE_SIZE = 10 * 1024 * 1024



def initialize_database():
    """Create the SQLite database and uploaded images table."""

    DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(DATABASE_PATH) as connection:

        connection.execute("""
            CREATE TABLE IF NOT EXISTS uploaded_images (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                original_filename TEXT NOT NULL,
                saved_filename TEXT NOT NULL,
                category TEXT NOT NULL,
                file_path TEXT NOT NULL,
                image_hash TEXT NOT NULL UNIQUE,
                status TEXT NOT NULL DEFAULT 'pending',
                uploaded_at TEXT NOT NULL
            )
        """)

        connection.commit()


def approve_uploaded_image(image_id):
    """Approve an uploaded image for future retraining."""

    initialize_database()

    with sqlite3.connect(DATABASE_PATH) as connection:

        cursor = connection.execute("""
            UPDATE uploaded_images
            SET status = 'approved'
            WHERE id = ?
              AND status = 'pending'
        """, (image_id,))

        if cursor.rowcount == 0:
            raise ValueError(
                f"Image ID {image_id} was not found or is not pending."
            )


def get_uploaded_images():
    """Return uploaded image records."""

    initialize_database()

    with sqlite3.connect(DATABASE_PATH) as connection:

        connection.row_factory = sqlite3.Row

        rows = connection.execute("""
            SELECT *
            FROM uploaded_images
            ORDER BY id DESC
        """).fetchall()

        return [dict(row) for row in rows]


def validate_image(contents):
    """Validate image size, format and image content."""

    if not contents:
        raise ValueError("The uploaded file is empty.")

    if len(contents) > MAX_FILE_SIZE:
        raise ValueError("Image exceeds the 10 MB upload limit.")

    try:
        with Image.open(BytesIO(contents)) as image:

            if image.format not in {"JPEG", "PNG", "WEBP"}:
                raise ValueError(
                    "Only JPEG, PNG and WebP images are supported."
                )

            image.verify()

    except (UnidentifiedImageError, OSError):
        raise ValueError("The uploaded file is not a valid image.")

    return True


def save_uploaded_image(contents, category, original_filename="uploaded_image"):
    """Save an uploaded image and its metadata in SQLite."""

    if category not in CLASS_NAMES:
        raise ValueError("Invalid image category.")

    validate_image(contents)

    initialize_database()

    image_hash = hashlib.sha256(contents).hexdigest()

    # Check for an existing image.
    with sqlite3.connect(DATABASE_PATH) as connection:

        existing = connection.execute("""
            SELECT saved_filename, category
            FROM uploaded_images
            WHERE image_hash = ?
        """, (image_hash,)).fetchone()

        if existing:

            if existing[1] != category:
                raise ValueError(
                    "This image already exists under a different category."
                )

            return existing[0]

    # Determine the image format and extension.
    with Image.open(BytesIO(contents)) as image:
        image_format = image.format

    extension = {
        "JPEG": ".jpg",
        "PNG": ".png",
        "WEBP": ".webp"
    }[image_format]

    # Create a unique filename.
    filename = f"{image_hash}{extension}"

    category_dir = UPLOAD_DIR / category
    category_dir.mkdir(parents=True, exist_ok=True)

    image_path = category_dir / filename

    # Save the actual image file.
    image_path.write_bytes(contents)

    try:
        with sqlite3.connect(DATABASE_PATH) as connection:

            connection.execute("""
                INSERT INTO uploaded_images (
                    original_filename,
                    saved_filename,
                    category,
                    file_path,
                    image_hash,
                    status,
                    uploaded_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                original_filename,
                filename,
                category,
                str(image_path),
                image_hash,
                "pending",
                datetime.now(timezone.utc).isoformat()
            ))

            connection.commit()

    except sqlite3.IntegrityError:

        # Remove the file if another upload inserted the same hash.
        if image_path.exists():
            image_path.unlink()

        raise ValueError("This image has already been uploaded.")

    return filename


def preprocess_image(image_path):
    """Convert an image to RGB, resize and normalize it."""

    try:
        with Image.open(image_path) as image:

            image = image.convert("RGB")
            image = image.resize(IMAGE_SIZE)

            image_array = np.asarray(
                image,
                dtype=np.float32
            ) / 255.0

        return image_array

    except (OSError, ValueError) as error:

        raise ValueError(
            f"Could not preprocess image {image_path}: {error}"
        ) from error



def prepare_training_dataset(include_uploaded=True):
    """
    Collect image paths and labels from original training data
    and approved uploaded images.

    Images are loaded later in batches using tf.data.
    """

    image_paths = []
    labels = []

    summary = {}
    seen_images = set()

    class_indices = {
        category: index
        for index, category in enumerate(CLASS_NAMES)
    }

    # Load original training image paths.
    for category in CLASS_NAMES:

        category_dir = TRAIN_DIR / category

        if not category_dir.exists():
            raise FileNotFoundError(
                f"Missing training folder: {category_dir}"
            )

        class_count = 0

        for image_path in sorted(category_dir.iterdir()):

            if not image_path.is_file():
                continue

            image_array = preprocess_image(image_path)

            image_hash = hashlib.sha256(
                image_array.tobytes()
            ).hexdigest()

            if image_hash in seen_images:
                print(f"Skipping duplicate: {image_path}")
                continue

            seen_images.add(image_hash)

            image_paths.append(str(image_path))
            labels.append(class_indices[category])

            class_count += 1

        summary[category] = class_count

    # Load approved uploaded images.
    uploaded_counts = {
        category: 0
        for category in CLASS_NAMES
    }

    if include_uploaded:

        initialize_database()

        with sqlite3.connect(DATABASE_PATH) as connection:

            connection.row_factory = sqlite3.Row

            uploaded_records = connection.execute("""
                SELECT *
                FROM uploaded_images
                WHERE status = 'approved'
                ORDER BY id
            """).fetchall()

        for record in uploaded_records:

            category = record["category"]
            image_path = Path(record["file_path"])

            if category not in CLASS_NAMES:
                continue

            if not image_path.is_file():
                print(f"Missing uploaded image: {image_path}")
                continue

            image_array = preprocess_image(image_path)

            image_hash = hashlib.sha256(
                image_array.tobytes()
            ).hexdigest()

            if image_hash in seen_images:
                print(f"Skipping duplicate upload: {image_path}")
                continue

            seen_images.add(image_hash)

            image_paths.append(str(image_path))
            labels.append(class_indices[category])

            uploaded_counts[category] += 1

    X = np.asarray(image_paths)
    y = np.asarray(labels, dtype=np.int32)

    if len(X) == 0:
        raise ValueError("No valid training images were found.")

    if set(np.unique(y)) != set(range(len(CLASS_NAMES))):
        raise ValueError(
            "The training dataset must contain all six classes."
        )

    print("\nTRAINING DATASET SUMMARY")
    print("------------------------")

    for category in CLASS_NAMES:
        print(
            f"{category}: {summary[category]} original, "
            f"{uploaded_counts[category]} uploaded"
        )

    print("Total images:", len(X))

    return X, y, summary


def split_retraining_dataset(X, y, validation_size=0.2):

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=validation_size,
        random_state=42,
        stratify=y
    )

    return X_train, X_val, y_train, y_val


def create_tf_dataset(
    image_paths,
    labels,
    batch_size=16,
    shuffle=False,
    class_weights=None,
    include_sample_weights=False
):
    """Load and preprocess images in batches."""

    dataset = tf.data.Dataset.from_tensor_slices(
        (image_paths, labels)
    )

    if shuffle:
        dataset = dataset.shuffle(
            buffer_size=len(image_paths),
            reshuffle_each_iteration=True
        )

    def load_image(image_path, label):

        image = tf.io.read_file(image_path)

        image = tf.io.decode_image(
            image,
            channels=3,
            expand_animations=False
        )

        image.set_shape([None, None, 3])

        image = tf.image.resize(
            image,
            IMAGE_SIZE
        )

        image = tf.cast(image, tf.float32) / 255.0

        return image, label

    dataset = dataset.map(
        load_image,
        num_parallel_calls=tf.data.AUTOTUNE
    )

    if include_sample_weights:

        if class_weights is None:
            raise ValueError(
                "class_weights must be provided when include_sample_weights is True"
            )

        # Labels are integer indices into CLASS_NAMES. TensorFlow cannot turn
        # the class-id-to-weight mapping into a constant directly, so build a
        # dense vector that can be indexed by each label in the dataset map.
        if isinstance(class_weights, dict):
            class_weight_values = [
                class_weights.get(class_index, 1.0)
                for class_index in range(len(CLASS_NAMES))
            ]
        else:
            class_weight_values = class_weights

        weights = tf.constant(class_weight_values, dtype=tf.float32)

        def add_sample_weight(image, label):
            sample_weight = tf.gather(
                weights,
                label
            )

            return image, label, sample_weight

        dataset = dataset.map(
            add_sample_weight,
            num_parallel_calls=tf.data.AUTOTUNE
        )

    dataset = dataset.batch(batch_size)

    return dataset.prefetch(1)



class MacroF1Checkpoint(Callback):

    def __init__(self, validation_data, save_path):
        super().__init__()

        self.X_val, self.y_val = validation_data
        self.save_path = Path(save_path)
        self.best_macro_f1 = -1.0

    def on_epoch_end(self, epoch, logs=None):

        probabilities = self.model.predict(
            self.X_val,
            verbose=0
        )

        predictions = np.argmax(probabilities, axis=1)

        macro_f1 = f1_score(
            self.y_val,
            predictions,
            average="macro",
            zero_division=0
        )

        print(f"\nValidation Macro-F1: {macro_f1:.4f}")

        if macro_f1 > self.best_macro_f1:

            self.best_macro_f1 = macro_f1

            self.model.save(self.save_path)

            print("New best candidate model saved!")



def configure_fine_tuning(model_path=MODEL_PATH):

    model = load_model(model_path)

    base_model = model.layers[2]

    base_model.trainable = True

    for index, layer in enumerate(base_model.layers):

        layer.trainable = index >= 134

        if "BatchNormalization" in layer.__class__.__name__:
            layer.trainable = False

    model.compile(
        optimizer=Adam(learning_rate=1e-5),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


class MacroF1Checkpoint(Callback):

    def __init__(self, validation_images, y_val, save_path):
        super().__init__()

        self.validation_images = validation_images
        self.y_val = y_val

        self.save_path = Path(save_path)
        self.best_macro_f1 = -1.0

    def on_epoch_end(self, epoch, logs=None):

        probabilities = self.model.predict(
            self.validation_images,
            verbose=0
        )

        predictions = np.argmax(probabilities, axis=1)

        macro_f1 = f1_score(
            self.y_val,
            predictions,
            average="macro",
            zero_division=0
        )

        print(f"\nValidation Macro-F1: {macro_f1:.4f}")

        if macro_f1 > self.best_macro_f1:

            self.best_macro_f1 = macro_f1

            self.model.save(self.save_path)

            print("New best candidate model saved!")

def retrain_model(
    X_train,
    y_train,
    X_val,
    y_val,
    model_path=MODEL_PATH,
    candidate_path=CANDIDATE_MODEL_PATH
):

    model = configure_fine_tuning(model_path)

    classes = np.unique(y_train)

    weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=y_train
    )

    class_weights = {
        int(class_index): float(weight)
        for class_index, weight in zip(classes, weights)
    }

    print("\nCLASS WEIGHTS")
    print(class_weights)

    # Create batched training dataset.
    train_dataset = create_tf_dataset(
        X_train,
        y_train,
        batch_size=16,
        shuffle=True,
        class_weights=class_weights,
        include_sample_weights=True
    )

    # Create batched validation dataset.
    validation_dataset = create_tf_dataset(
        X_val,
        y_val,
        batch_size=16
    )

    # Create image-only dataset for macro-F1 evaluation.
    validation_images = tf.data.Dataset.from_tensor_slices(
        X_val
    )

    def load_validation_image(image_path):

        image = tf.io.read_file(image_path)

        image = tf.io.decode_image(
            image,
            channels=3,
            expand_animations=False
        )

        image.set_shape([None, None, 3])

        image = tf.image.resize(
            image,
            IMAGE_SIZE
        )

        return tf.cast(image, tf.float32) / 255.0

    validation_images = validation_images.map(
        load_validation_image,
        num_parallel_calls=tf.data.AUTOTUNE
    ).batch(16).prefetch(1)

    candidate_path = Path(candidate_path)

    candidate_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    if candidate_path.exists():
        candidate_path.unlink()

    macro_f1_checkpoint = MacroF1Checkpoint(
        validation_images=validation_images,
        y_val=y_val,
        save_path=candidate_path
    )

    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=3,
        restore_best_weights=True
    )

    reduce_lr = ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.2,
        patience=2,
        min_lr=1e-7
    )

    history = model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=10,
        callbacks=[
            macro_f1_checkpoint,
            early_stopping,
            reduce_lr
        ],
        verbose=1
    )

    if not candidate_path.is_file():
        raise RuntimeError(
            "Training finished without saving a candidate model."
        )

    print("\nRETRAINING COMPLETED")

    print(
        "Best validation macro-F1:",
        macro_f1_checkpoint.best_macro_f1
    )

    print("Candidate model:", candidate_path)

    return history, macro_f1_checkpoint.best_macro_f1
