"""
Feature Extraction Module
--------------------------
Prepares a cropped face image for CNN input: resizing, normalization,
and optional contrast enhancement (CLAHE) to make subtle biomarkers
such as pallor or pigmentation changes more distinguishable.

Corresponds to Chapter 7.4 of the NUTRICARE report.
"""

import cv2
import numpy as np

IMAGE_SIZE = (224, 224)  # matches the CNN input in Chapter 7.5.1


def enhance_contrast(image_bgr: np.ndarray) -> np.ndarray:
    """Apply CLAHE on the L channel (LAB colour space) to boost local contrast
    without distorting colour, which helps highlight pallor/pigmentation cues."""
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_eq = clahe.apply(l)
    lab_eq = cv2.merge((l_eq, a, b))
    return cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)


def preprocess_face(face_bgr: np.ndarray, apply_clahe: bool = True) -> np.ndarray:
    """
    Resize to the CNN's expected input dimensions and normalize pixel
    values to [0, 1]. Returns an RGB float32 array of shape (224, 224, 3).
    """
    if apply_clahe:
        face_bgr = enhance_contrast(face_bgr)

    resized = cv2.resize(face_bgr, IMAGE_SIZE, interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    normalized = rgb.astype(np.float32) / 255.0
    return normalized


def to_batch(preprocessed_image: np.ndarray) -> np.ndarray:
    """Add the batch dimension expected by model.predict()."""
    return np.expand_dims(preprocessed_image, axis=0)


def build_augmentation_layer():
    """
    Returns a Keras Sequential of augmentation layers used only during
    training (rotation, brightness, horizontal flip) as described in
    Chapter 7.4. Imported lazily so this module has no hard TF dependency
    at inference time in environments where only OpenCV is available.
    """
    from tensorflow.keras import layers, Sequential

    return Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.05),
        layers.RandomBrightness(0.15),
        layers.RandomContrast(0.15),
    ], name="augmentation")
