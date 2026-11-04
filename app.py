
import json
import numpy as np
import streamlit as st
from PIL import Image
from pathlib import Path
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# ---------------- PAGE CONFIGURATION ----------------
st.set_page_config(
    page_title="AI-SORT | Waste Classification",
    page_icon="♻️",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ---------------- FILE PATHS ----------------
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "waste_classifier_v2.keras"
CLASS_PATH = BASE_DIR / "models" / "class_names_v2.json"

# ---------------- COMPACT CSS ----------------
st.markdown("""
<style>
.block-container {
    max-width: 950px;
    padding-top: 2.5rem !important;
    padding-bottom: 0.5rem !important;
    padding-left: 1rem;
    padding-right: 1rem;
}z

div[data-testid="stVerticalBlock"] {
    gap: 0.6rem;
}

div[data-testid="stImage"] img {
    max-height: 240px;
    object-fit: contain;
}


div.stButton > button {
    width: 220px;
    min-height: 2.2rem;
    border-radius: 8px;
    font-weight: 600;
    display: block;
    margin: 0 auto;
}


/* Full-width, compact upload box */
div[data-testid="stFileUploader"] {
    width: 100%;
    max-width: 100%;
    margin: 0;
}

div[data-testid="stFileUploader"] section {
    padding: 0.35rem 0.6rem !important;
}

div[data-testid="stFileUploader"] [data-testid="stFileUploaderDropzone"] {
    min-height: 70px !important;
    padding: 0.5rem !important;
    border-radius: 8px;
}


.footer {
    text-align: center;
    font-size: 12px;
    color: #888888;
    margin-top: 12px;
}
</style>
""", unsafe_allow_html=True)

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
    return tf.keras.models.load_model(MODEL_PATH)

@st.cache_data
def load_class_names():
    if not CLASS_PATH.exists():
        raise FileNotFoundError(f"Class names file not found: {CLASS_PATH}")

    with open(CLASS_PATH, "r", encoding="utf-8") as file:
        names = json.load(file)

    if isinstance(names, dict):
        names = [
            names[str(i)] if str(i) in names else names[i]
            for i in range(len(names))
        ]

    return names

try:
    model = load_model()
    class_names = load_class_names()

    if len(class_names) != model.output_shape[-1]:
        st.error("Class names do not match the model output.")
        st.stop()

except Exception as error:
    st.error(f"Unable to load the model: {error}")
    st.stop()

# ---------------- HEADER ----------------
st.markdown("""
<div style="
    text-align:center;
    font-size:36px;
    font-weight:800;
    line-height:1.4;
    padding-bottom:4px;
    color:#21c55d;">
    ♻️ AI-SORT
</div>
<div style="
    text-align:center;
    font-size:14px;
    margin-bottom:10px;
    color:#aaaaaa;">
    Intelligent Waste Classification Using Deep Learning
</div>
""", unsafe_allow_html=True)

# ---------------- UPLOAD SECTION ----------------
st.markdown("### Upload Waste Image")

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"],
    label_visibility="collapsed"
)

# ---------------- DISPOSAL GUIDANCE ----------------
disposal_advice = {
    "glass": "Recycle in a suitable glass recycling bin.",
    "metal": "Recycle in a suitable metal recycling bin.",
    "organic": "Compost or place in the organic waste bin.",
    "paper": "Keep clean and dry for paper recycling.",
    "plastic": "Place in the appropriate plastic recycling bin."
}

# ---------------- PREDICTION ----------------
if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert("RGB")

        if st.button("🔍 Classify Waste", use_container_width=True):
            with st.spinner("Analyzing image..."):
                resized_image = image.resize((224, 224))

                image_array = np.array(
                    resized_image,
                    dtype=np.float32
                )

                # MobileNetV2 preprocessing: pixel values to [-1, 1]
                image_array = preprocess_input(image_array)
                image_array = np.expand_dims(image_array, axis=0)

                predictions = model.predict(
                    image_array,
                    verbose=0
                )[0]

                predicted_index = int(np.argmax(predictions))
                predicted_class = str(class_names[predicted_index])
                confidence = float(predictions[predicted_index]) * 100

                st.session_state["waste_result"] = {
                    "image": image.copy(),
                    "category": predicted_class,
                    "confidence": confidence
                }

        # Clear old result when a different image is uploaded
        result = st.session_state.get("waste_result")

        if result is not None:
            left, right = st.columns([1, 1], gap="medium")

            with left:
                st.image(
                    result["image"],
                    caption="Uploaded Waste Image",
                    width=260
                )

            with right:
                st.markdown("### Prediction Result")
                st.success(f"Category: {result['category'].title()}")

                st.metric(
                    "Model Confidence",
                    f"{result['confidence']:.2f}%"
                )

                advice = disposal_advice.get(
                    result["category"].strip().lower(),
                    "Follow your local waste disposal guidelines."
                )

                st.caption("Disposal Advice")
                st.write(advice)

    except Exception as error:
        st.error(f"Could not process this image: {error}")

else:
    st.session_state.pop("waste_result", None)
    st.info("Upload a JPG or PNG image to classify the waste.")

# ---------------- FOOTER ----------------
st.markdown(
    '<div class="footer">'
    'AI-SORT | AI-Based Waste Classification System'
    '</div>',
    unsafe_allow_html=True
)
