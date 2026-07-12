"""
Image Acquisition / Validation Module
--------------------------------------
Checks resolution, blur, and lighting before an image is accepted for
face detection and CNN analysis. Corresponds to Chapter 7.2 of the
NUTRICARE report (Image Acquisition Module) and FR/NFR in Chapter 5.
"""

import cv2
import numpy as np


class ImageValidationResult:
    def __init__(self, ok: bool, reason: str = ""):
        self.ok = ok
        self.reason = reason

    def __bool__(self):
        return self.ok

    def __repr__(self):
        return f"ImageValidationResult(ok={self.ok}, reason='{self.reason}')"


def validate_image(
    image_bgr: np.ndarray,
    min_width: int = 200,
    min_height: int = 200,
    blur_threshold: float = 60.0,
    dark_threshold: float = 40.0,
    bright_threshold: float = 220.0,
) -> ImageValidationResult:
    """
    Runs three checks used by the report's TC-02 / TC-04 test cases:
      1. Minimum resolution
      2. Blur detection (variance of Laplacian)
      3. Lighting (mean pixel brightness within an acceptable band)
    """
    if image_bgr is None or image_bgr.size == 0:
        return ImageValidationResult(False, "Empty or unreadable image.")

    h, w = image_bgr.shape[:2]
    if w < min_width or h < min_height:
        return ImageValidationResult(
            False, f"Image resolution too low ({w}x{h}); minimum is {min_width}x{min_height}."
        )

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    # Blur check: low variance of the Laplacian indicates a blurry image.
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    if laplacian_var < blur_threshold:
        return ImageValidationResult(
            False, f"Image appears blurry (sharpness score {laplacian_var:.1f})."
        )

    # Lighting check: mean brightness should not be too dark or too washed out.
    mean_brightness = float(np.mean(gray))
    if mean_brightness < dark_threshold:
        return ImageValidationResult(
            False, f"Image is too dark (mean brightness {mean_brightness:.1f})."
        )
    if mean_brightness > bright_threshold:
        return ImageValidationResult(
            False, f"Image is overexposed (mean brightness {mean_brightness:.1f})."
        )

    return ImageValidationResult(True, "Image passed quality checks.")
