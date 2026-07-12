"""
Unit and integration tests mirroring the test cases described in
Chapter 7.9.3 of the NUTRICARE report (TC-01 .. TC-06 subset that
applies to the offline pipeline; TC-07..TC-10 relate to deployment
infrastructure and are out of scope for local unit tests).

Run with:
    python -m pytest test_pipeline.py -v
"""

import numpy as np
import cv2
import pytest

from modules.image_validation import validate_image
from modules.face_detection import FaceDetector, FaceDetectionError
from modules.feature_extraction import preprocess_face, IMAGE_SIZE
from modules.recommendation import get_recommendation, RECOMMENDATIONS


def make_blank_image(w=300, h=300, color=(120, 120, 120)):
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:] = color
    # add a little texture/noise so this doesn't get flagged as "blurry"
    # (zero-variance flat colour trips the Laplacian-variance blur check)
    rng = np.random.default_rng(0)
    noise = rng.integers(-8, 8, size=img.shape, dtype=np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return img


def make_face_like_image(size=300):
    from generate_demo_dataset import make_synthetic_face
    return make_synthetic_face(size=size, seed=1)


# TC-02 equivalent: low resolution image should fail validation
def test_validate_image_rejects_low_resolution():
    tiny = make_blank_image(w=50, h=50)
    result = validate_image(tiny)
    assert not result.ok
    assert "resolution" in result.reason.lower()


# TC-04 equivalent: very dark image should fail validation
def test_validate_image_rejects_dark_image():
    dark = make_blank_image(color=(5, 5, 5))
    result = validate_image(dark)
    assert not result.ok
    assert "dark" in result.reason.lower()


def test_validate_image_accepts_good_image():
    good = make_face_like_image()
    result = validate_image(good)
    assert result.ok


# TC-03 equivalent: image with no detectable face
def test_face_detection_raises_on_blank_image():
    blank = make_blank_image()
    detector = FaceDetector()
    with pytest.raises(FaceDetectionError):
        detector.get_largest_face(blank)


def test_preprocess_face_output_shape():
    face = make_face_like_image()
    processed = preprocess_face(face)
    assert processed.shape == (IMAGE_SIZE[0], IMAGE_SIZE[1], 3)
    assert processed.dtype == np.float32
    assert processed.max() <= 1.0 and processed.min() >= 0.0


def test_recommendation_lookup_known_classes():
    for cls in RECOMMENDATIONS.keys():
        rec = get_recommendation(cls)
        assert "summary" in rec
        assert isinstance(rec["foods"], list)


def test_recommendation_lookup_unknown_class_has_safe_fallback():
    rec = get_recommendation("Not_A_Real_Class")
    assert "consult" in " ".join(rec["tips"]).lower()
