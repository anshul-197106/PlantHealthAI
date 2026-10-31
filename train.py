from pathlib import Path
import json

import tensorflow as tf
from tensorflow.keras import layers

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models"
IMG_SIZE = (224, 224)
SEED = 42


def load_split(name, shuffle):
    folder = DATA_DIR / name
    if not folder.exists():
        raise FileNotFoundError(f"{folder} not found. Run prepare_dataset.py first.")
    return tf.keras.utils.image_dataset_from_directory(folder, image_size=IMG_SIZE, batch_size=32, label_mode="int", shuffle=shuffle, seed=SEED)


def main():
    tf.keras.utils.set_random_seed(SEED)
    train_ds = load_split("train", True)
    classes = train_ds.class_names
    val_ds = load_split("val", False)
    if val_ds.class_names != classes:
        raise ValueError("Train/validation class names do not match.")
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    with (MODEL_DIR / "classes.json").open("w", encoding="utf-8") as f:
        json.dump(classes, f, indent=2)
    auto = tf.data.AUTOTUNE
    train_ds, val_ds = train_ds.prefetch(auto), val_ds.prefetch(auto)
    augment = tf.keras.Sequential([layers.RandomFlip("horizontal"), layers.RandomRotation(0.08), layers.RandomZoom(0.1)], name="augmentation")
    base = tf.keras.applications.MobileNetV2(input_shape=(*IMG_SIZE, 3), include_top=False, weights="imagenet")
    base.trainable = False
    inputs = layers.Input(shape=(*IMG_SIZE, 3))
    x = augment(inputs)
    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.25)(x)
    outputs = layers.Dense(len(classes), activation="softmax")(x)
    model = tf.keras.Model(inputs, outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    callbacks = [tf.keras.callbacks.ModelCheckpoint(str(MODEL_DIR / "best_model.keras"), monitor="val_loss", save_best_only=True), tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True)]
    model.fit(train_ds, validation_data=val_ds, epochs=8, callbacks=callbacks)
    model.save(MODEL_DIR / "plant_disease_model.keras")
    print(f"Saved model and {len(classes)} classes under {MODEL_DIR}")


if __name__ == "__main__":
    main()
