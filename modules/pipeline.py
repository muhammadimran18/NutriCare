"""
NUTRICARE Pipeline
--------------------
Orchestrates the full flow described in Chapter 6 (System Architecture)
and Chapter 6.3.1 (Workflow Sequence):

  Upload Image -> Validate -> Face Detection -> Pre-processing
  -> CNN Analysis -> Deficiency Prediction -> Recommendation -> Result
"""

import os
import numpy as np

from .image_validation import validate_image
from .face_detection import FaceDetector, FaceDetectionError
from .feature_extraction import preprocess_face, to_batch
from .cnn_model import CLASS_NAMES, load_trained_model
from .recommendation import get_recommendation


class PipelineResult:
    def __init__(self, success, message, predicted_class=None,
                 confidence=None, probabilities=None, recommendation=None,
                 face_crop=None):
        self.success = success
        self.message = message
        self.predicted_class = predicted_class
        self.confidence = confidence
        self.probabilities = probabilities
        self.recommendation = recommendation
        self.face_crop = face_crop

    def to_dict(self):
        return {
            "success": self.success,
            "message": self.message,
            "predicted_class": self.predicted_class,
            "confidence": self.confidence,
            "probabilities": self.probabilities,
            "recommendation": self.recommendation,
        }


class NutricarePipeline:
    def __init__(self, model_path: str):
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"No trained model found at '{model_path}'. Run train.py first."
            )
        self.model = load_trained_model(model_path)
        self.detector = FaceDetector()

    def run(self, image_bgr: np.ndarray) -> PipelineResult:
        # Step 1: Validate image quality (TC-02, TC-04)
        validation = validate_image(image_bgr)
        if not validation.ok:
            return PipelineResult(False, validation.reason)

        # Step 2: Face detection (TC-03)
        try:
            face_crop = self.detector.crop_and_align(image_bgr)
        except FaceDetectionError as e:
            return PipelineResult(False, str(e))

        # Step 3: Feature extraction / pre-processing
        preprocessed = preprocess_face(face_crop)
        batch = to_batch(preprocessed)

        # Step 4: CNN classification
        probs = self.model.predict(batch, verbose=0)[0]
        class_idx = int(np.argmax(probs))
        predicted_class = CLASS_NAMES[class_idx]
        confidence = float(probs[class_idx])
        prob_dict = {name: float(p) for name, p in zip(CLASS_NAMES, probs)}

        # Step 5: Recommendation generation
        recommendation = get_recommendation(predicted_class)

        return PipelineResult(
            success=True,
            message="Prediction complete.",
            predicted_class=predicted_class,
            confidence=confidence,
            probabilities=prob_dict,
            recommendation=recommendation,
            face_crop=face_crop,
        )
