# NUTRICARE — AI-Based Nutritional Deficiency Detection Using Facial Image Analysis

Working implementation of the pipeline described in the accompanying project
report: **Upload/Capture Image → Validate → Face Detection → Pre-processing →
CNN Classification → Recommendation → Result**.

> ⚠️ **Screening tool only.** This is a student/academic prototype, not a
> medical device. It must never be used to make real clinical decisions
> without a properly collected, labelled, consented medical dataset and
> proper clinical validation.

## Project structure

```
nutricare/
├── app.py                     # Streamlit web interface (Presentation Layer)
├── train.py                   # CNN training script
├── generate_demo_dataset.py   # Creates a synthetic dataset for testing
├── test_pipeline.py           # Unit / integration tests (pytest)
├── requirements.txt
├── modules/
│   ├── image_validation.py    # Ch 7.2 — Image Acquisition / quality checks
│   ├── face_detection.py      # Ch 7.3 — Face Detection Module (OpenCV)
│   ├── feature_extraction.py  # Ch 7.4 — Feature Extraction Module
│   ├── cnn_model.py            # Ch 7.5 — CNN Classification Module
│   ├── recommendation.py      # Ch 7.6 — Recommendation Module
│   └── pipeline.py            # Orchestrates the full workflow (Ch 6)
├── data/dataset/              # Training images, one sub-folder per class
└── models/                    # Saved trained model (.keras)
```

## Quick start

```bash
pip install -r requirements.txt

# 1. Generate a small synthetic dataset so the pipeline can be exercised
#    end-to-end without a real clinical dataset (see warning below).
python generate_demo_dataset.py --per_class 40

# 2. Train the CNN (baseline architecture from report Table 7.5.1)
python train.py --data_dir data/dataset --epochs 10 --arch baseline
# or, for better accuracy on small datasets, a MobileNetV2 transfer model:
python train.py --data_dir data/dataset --epochs 10 --arch transfer

# 3. Run the tests
python -m pytest test_pipeline.py -v

# 4. Launch the web app
streamlit run app.py
```

## ⚠️ About the synthetic demo dataset

`generate_demo_dataset.py` draws simple cartoon-style faces with a
class-dependent colour tint (a stand-in for pallor/pigmentation cues) purely
so that `train.py`, the CNN, and the recommendation module can be exercised
mechanically without a real dataset. Two things follow from that:

1. **It is enough to train and validate the CNN classification module** —
   in local testing the baseline CNN reached ~87% validation accuracy on
   the synthetic classes in a few epochs, confirming the training loop,
   model architecture, and inference code all work correctly.
2. **It is *not* enough for the Face Detection Module.** OpenCV's Haar
   cascade is trained on real photographic faces, so it correctly does
   **not** treat the cartoon renders as faces (this is validated in
   `test_pipeline.py`). To see the full pipeline (including face detection)
   work end-to-end, replace `data/dataset/<class>/` with real, consented,
   appropriately labelled facial photographs, and use real photos when
   testing `app.py` or `modules/pipeline.py`.

## Replacing the demo data with a real dataset

Put real images into the same folder layout:

```
data/dataset/
    Iron/
    Vitamin_B12/
    Vitamin_D/
    Protein/
    No_Deficiency/
```

Then re-run `train.py`. No code changes are required — `image_dataset_from_directory`
infers classes from folder names, and `CLASS_NAMES` in `modules/cnn_model.py`
already matches this taxonomy.

## Running tests

`test_pipeline.py` covers the module-level test cases from Chapter 7.9.3 of
the report (TC‑01 to TC‑06 equivalents): resolution/blur/lighting validation,
face-detection failure handling, feature-extraction output shape, and
recommendation lookups (including the unknown-class fallback).

```bash
python -m pytest test_pipeline.py -v
```

## Notes on the CNN architecture

`modules/cnn_model.py` provides two options:

- `build_baseline_cnn()` — the exact architecture from report Table 7.5.1
  (Conv32 → Conv64 → Conv128 → Dropout 0.5 → Dense128 → Dense5/softmax).
- `build_transfer_learning_model()` — a MobileNetV2-backed classifier for
  better generalization on small real-world datasets, as recommended in
  Chapter 7.5 (transfer learning with VGG/ResNet-style backbones).
