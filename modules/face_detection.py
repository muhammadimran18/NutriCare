"""
Face Detection Module
----------------------
Locates the facial region in an input image using OpenCV's Haar-cascade
detector, then crops and aligns it so that downstream modules always
receive a consistent, face-centred image.

This corresponds to Chapter 7.3 of the NUTRICARE project report.
"""

import cv2
import numpy as np


class FaceDetectionError(Exception):
    """Raised when no face can be located in the supplied image."""


class FaceDetector:
    def __init__(self, min_face_size=(80, 80)):
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.cascade = cv2.CascadeClassifier(cascade_path)
        if self.cascade.empty():
            raise RuntimeError("Could not load Haar cascade for face detection.")
        self.min_face_size = min_face_size

    def detect_faces(self, image_bgr: np.ndarray):
        """Return a list of (x, y, w, h) bounding boxes for detected faces."""
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)  # improves detection under uneven lighting
        faces = self.cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=self.min_face_size,
        )
        return faces

    def get_largest_face(self, image_bgr: np.ndarray):
        """Return bounding box of the largest detected face (assumed primary subject)."""
        faces = self.detect_faces(image_bgr)
        if len(faces) == 0:
            raise FaceDetectionError("No face detected in the supplied image.")
        # pick the largest bounding box by area
        largest = max(faces, key=lambda f: f[2] * f[3])
        return largest

    def crop_and_align(self, image_bgr: np.ndarray, margin: float = 0.25) -> np.ndarray:
        """
        Crop the facial region with a margin around it (so chin/forehead
        aren't clipped), returning a BGR image ready for pre-processing.
        """
        x, y, w, h = self.get_largest_face(image_bgr)

        mx, my = int(w * margin), int(h * margin)
        x0 = max(x - mx, 0)
        y0 = max(y - my, 0)
        x1 = min(x + w + mx, image_bgr.shape[1])
        y1 = min(y + h + my, image_bgr.shape[0])

        face_crop = image_bgr[y0:y1, x0:x1]
        return face_crop
