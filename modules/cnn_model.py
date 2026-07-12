"""
CNN Classification Module
---------------------------
Defines the NUTRICARE CNN architecture exactly as specified in
Chapter 7.5.1 of the project report, plus a transfer-learning variant
(MobileNetV2 backbone) for better accuracy on small datasets, as
recommended in Chapter 7.5.

Classes (5): Iron, Vitamin B12, Vitamin D, Protein, No_Deficiency
"""

from tensorflow import keras
from tensorflow.keras import layers, models

CLASS_NAMES = ["Iron", "Vitamin_B12", "Vitamin_D", "Protein", "No_Deficiency"]
INPUT_SHAPE = (224, 224, 3)


def build_baseline_cnn(input_shape=INPUT_SHAPE, num_classes=len(CLASS_NAMES)) -> keras.Model:
    """Builds the representative CNN architecture from report Table 7.5.1."""
    model = models.Sequential([
        layers.Input(shape=input_shape),

        layers.Conv2D(32, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(64, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(128, (3, 3), activation="relu"),
        layers.Dropout(0.5),

        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dense(num_classes, activation="softmax"),
    ], name="nutricare_baseline_cnn")

    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_transfer_learning_model(
    input_shape=INPUT_SHAPE, num_classes=len(CLASS_NAMES), fine_tune=False
) -> keras.Model:
    """
    Builds a MobileNetV2-backed classifier. MobileNetV2 is used (rather
    than the heavier VGG/ResNet mentioned in the report) because it is
    small enough to fine-tune quickly on limited medical-image datasets
    while still transferring strong low-level visual features.
    """
    base = keras.applications.MobileNetV2(
        input_shape=input_shape, include_top=False, weights="imagenet"
    )
    base.trainable = fine_tune

    inputs = keras.Input(shape=input_shape)
    x = keras.applications.mobilenet_v2.preprocess_input(inputs * 255.0)
    x = base(x, training=fine_tune)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, activation="relu")(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = keras.Model(inputs, outputs, name="nutricare_transfer_cnn")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-4),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def load_trained_model(model_path: str) -> keras.Model:
    return keras.models.load_model(model_path)
