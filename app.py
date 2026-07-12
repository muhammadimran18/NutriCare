"""
NUTRICARE Streamlit Application
----------------------------------
Presentation layer described in Chapter 6.1.1 (Presentation Layer).
Lets a user upload a facial photo, runs it through the full pipeline,
and displays the predicted deficiency, confidence score, and dietary
recommendation.

Run with:
    streamlit run app.py
"""

import numpy as np
import cv2
import streamlit as st

from modules.pipeline import NutricarePipeline

MODEL_PATH = "models/nutricare_model.keras"

st.set_page_config(page_title="NUTRICARE", page_icon="🥗", layout="centered")


@st.cache_resource
def get_pipeline():
    return NutricarePipeline(MODEL_PATH)


def read_image_file(uploaded_file) -> np.ndarray:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    return image_bgr


def main():
    st.title("🥗 NUTRICARE")
    st.caption("AI-Based Nutritional Deficiency Detection Using Facial Image Analysis")

    st.warning(
        "Screening tool only — not a medical diagnosis. Please consult a "
        "healthcare professional for confirmatory testing.",
        icon="⚠️",
    )

    uploaded_file = st.file_uploader(
        "Upload a clear, well-lit, front-facing photo", type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        image_bgr = read_image_file(uploaded_file)

        if image_bgr is None:
            st.error("Could not read the uploaded file as an image.")
            return

        st.image(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB), caption="Uploaded image", width=300)

        try:
            pipeline = get_pipeline()
        except FileNotFoundError as e:
            st.error(str(e))
            st.info("Run `python generate_demo_dataset.py` then `python train.py` first.")
            return

        with st.spinner("Analyzing image..."):
            result = pipeline.run(image_bgr)

        if not result.success:
            st.error(result.message)
            return

        st.image(
            cv2.cvtColor(result.face_crop, cv2.COLOR_BGR2RGB),
            caption="Detected face region",
            width=200,
        )

        st.subheader("Prediction")
        st.metric("Deficiency Category", result.predicted_class.replace("_", " "))
        st.progress(result.confidence)
        st.write(f"Confidence: **{result.confidence * 100:.1f}%**")

        with st.expander("Full probability breakdown"):
            for cls, prob in sorted(result.probabilities.items(), key=lambda x: -x[1]):
                st.write(f"{cls.replace('_', ' ')}: {prob * 100:.1f}%")

        st.subheader("Recommendation")
        rec = result.recommendation
        st.write(rec["summary"])
        if rec["foods"]:
            st.write("**Suggested foods:**")
            for food in rec["foods"]:
                st.write(f"- {food}")
        if rec["tips"]:
            st.write("**Tips:**")
            for tip in rec["tips"]:
                st.write(f"- {tip}")


if __name__ == "__main__":
    main()
