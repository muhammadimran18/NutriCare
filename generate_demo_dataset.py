"""
Generates a small synthetic demo dataset so the full training/inference
pipeline can be exercised end-to-end without a real clinical dataset.

IMPORTANT: This produces synthetic face-like images with class-dependent
colour tints (a stand-in for pallor/pigmentation cues) -- it is only
meant to prove the pipeline works mechanically. Before any real-world
use, replace data/dataset/<class>/ with genuine, ethically-sourced,
consented, and clinically-labelled facial images.

Usage:
    python generate_demo_dataset.py --per_class 40
"""

import argparse
import os
import numpy as np
import cv2

from modules.cnn_model import CLASS_NAMES

# Rough tint/brightness bias per class, purely to give the synthetic
# images *some* learnable signal so training converges in the demo.
CLASS_BIAS = {
    "Iron": {"tint": (180, 170, 190), "brightness": 0.85},          # pale
    "Vitamin_B12": {"tint": (150, 190, 170), "brightness": 0.90},   # sallow/yellowish
    "Vitamin_D": {"tint": (200, 190, 180), "brightness": 0.80},     # dull
    "Protein": {"tint": (190, 175, 165), "brightness": 0.88},       # dry
    "No_Deficiency": {"tint": (200, 160, 140), "brightness": 1.05}, # healthy tone
}


def make_synthetic_face(size=260, tint=(200, 160, 140), brightness=1.0, seed=None):
    rng = np.random.default_rng(seed)
    canvas = np.ones((size, size, 3), dtype=np.uint8)
    base_color = np.array(tint, dtype=np.float32) * brightness
    base_color = np.clip(base_color, 0, 255).astype(np.uint8)
    canvas[:, :] = base_color

    center = (size // 2, size // 2)
    axes = (size // 3, int(size // 2.4))
    cv2.ellipse(canvas, center, axes, 0, 0, 360, tuple(int(c) for c in base_color), -1)

    # eyes
    eye_y = size // 2 - 20
    cv2.circle(canvas, (center[0] - 35, eye_y), 10, (40, 40, 40), -1)
    cv2.circle(canvas, (center[0] + 35, eye_y), 10, (40, 40, 40), -1)

    # nose
    cv2.line(canvas, (center[0], eye_y + 10), (center[0], eye_y + 45), (90, 90, 90), 2)

    # mouth
    cv2.ellipse(canvas, (center[0], eye_y + 70), (25, 10), 0, 0, 180, (80, 60, 60), 2)

    # random noise/texture so images aren't identical
    noise = rng.normal(0, 6, canvas.shape).astype(np.int16)
    noisy = np.clip(canvas.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return noisy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--per_class", type=int, default=40)
    parser.add_argument("--out_dir", type=str, default="data/dataset")
    args = parser.parse_args()

    for class_name in CLASS_NAMES:
        class_dir = os.path.join(args.out_dir, class_name)
        os.makedirs(class_dir, exist_ok=True)
        bias = CLASS_BIAS[class_name]
        for i in range(args.per_class):
            img = make_synthetic_face(
                tint=bias["tint"], brightness=bias["brightness"], seed=i
            )
            out_path = os.path.join(class_dir, f"{class_name}_{i:03d}.jpg")
            cv2.imwrite(out_path, img)
        print(f"Generated {args.per_class} images for class '{class_name}' -> {class_dir}")


if __name__ == "__main__":
    main()
