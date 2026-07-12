"""
Training script for the NUTRICARE CNN Classification Module.

Expects a dataset directory structured as:

    data/dataset/
        Iron/*.jpg
        Vitamin_B12/*.jpg
        Vitamin_D/*.jpg
        Protein/*.jpg
        No_Deficiency/*.jpg

Usage:
    python train.py --data_dir data/dataset --epochs 10 --arch baseline
    python train.py --data_dir data/dataset --epochs 10 --arch transfer
"""

import argparse
import os

import tensorflow as tf
from tensorflow import keras

from modules.cnn_model import (
    build_baseline_cnn,
    build_transfer_learning_model,
    CLASS_NAMES,
)
from modules.feature_extraction import IMAGE_SIZE


def build_datasets(data_dir, batch_size=16, validation_split=0.2, seed=42):
    train_ds = keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=validation_split,
        subset="training",
        seed=seed,
        image_size=IMAGE_SIZE,
        batch_size=batch_size,
        label_mode="categorical",
        class_names=CLASS_NAMES,
    )
    val_ds = keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=validation_split,
        subset="validation",
        seed=seed,
        image_size=IMAGE_SIZE,
        batch_size=batch_size,
        label_mode="categorical",
        class_names=CLASS_NAMES,
    )

    normalization = keras.layers.Rescaling(1.0 / 255)
    train_ds = train_ds.map(lambda x, y: (normalization(x), y))
    val_ds = val_ds.map(lambda x, y: (normalization(x), y))

    train_ds = train_ds.cache().shuffle(200).prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.cache().prefetch(tf.data.AUTOTUNE)
    return train_ds, val_ds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/dataset")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--arch", choices=["baseline", "transfer"], default="baseline")
    parser.add_argument("--out_path", type=str, default="models/nutricare_model.keras")
    args = parser.parse_args()

    train_ds, val_ds = build_datasets(args.data_dir, batch_size=args.batch_size)

    if args.arch == "baseline":
        model = build_baseline_cnn()
    else:
        model = build_transfer_learning_model()

    model.summary()

    os.makedirs(os.path.dirname(args.out_path), exist_ok=True)
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            args.out_path, save_best_only=True, monitor="val_accuracy", mode="max"
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=4, restore_best_weights=True
        ),
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=callbacks,
    )

    model.save(args.out_path)
    print(f"\nTraining complete. Model saved to: {args.out_path}")

    final_val_acc = history.history.get("val_accuracy", [None])[-1]
    print(f"Final validation accuracy: {final_val_acc}")


if __name__ == "__main__":
    main()
