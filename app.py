import os
from pathlib import Path

import streamlit as st
import torch

from tuberculosis_recognition.cnn_model import CNN
from tuberculosis_recognition.inference import ImageRecognition
from tuberculosis_recognition.paths import EXAMPLES_DIR, MODEL_WEIGHTS_PATH


device = "cuda" if torch.cuda.is_available() else "cpu"
weights = Path(os.getenv("MODEL_WEIGHTS_PATH", str(MODEL_WEIGHTS_PATH)))


@st.cache_resource
def load_recognizer(weights_path):
    model = CNN()
    model.load_state_dict(torch.load(weights_path, map_location=device))
    return ImageRecognition(model)


if not weights.exists():
    st.error(
        f"Model weights not found: {weights}. "
        "Upload model_weights.pth to source/model/ or set MODEL_WEIGHTS_PATH."
    )
    st.stop()

recognizer = load_recognizer(str(weights))

extensions = [
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".gif",
    ".tif",
    ".jp2",
    ".pcx",
    ".ppm",
    ".tga"
]

path_dict = {
    "Normal 1": EXAMPLES_DIR / "Normal" / "1.png",
    "Normal 2": EXAMPLES_DIR / "Normal" / "2.png",
    "Normal 3": EXAMPLES_DIR / "Normal" / "3.png",
    "Tuberculosis 1": EXAMPLES_DIR / "Tuberculosis" / "1.png",
    "Tuberculosis 2": EXAMPLES_DIR / "Tuberculosis" / "2.jpg",
    "Tuberculosis 3": EXAMPLES_DIR / "Tuberculosis" / "3.jpg"
}

m = st.markdown("""
<style>
div.stButton > button:first-child {
    height: 4em;
    width: 100%;
}

.top-links {
    display: flex;
    gap: 0.75rem;
    margin-bottom: 1rem;
}

.top-links a {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    min-width: 8rem;
    padding: 0.6rem 1rem;
    border: 1px solid rgba(49, 51, 63, 0.2);
    border-radius: 0.75rem;
    color: inherit;
    text-decoration: none;
    font-weight: 600;
    background: rgba(255, 255, 255, 0.7);
}

.top-links a:hover {
    border-color: rgba(49, 51, 63, 0.45);
}
</style>""", unsafe_allow_html=True)

st.markdown(
    """
    <div class="top-links">
        <a href="https://github.com/PandaMia/Tuberculosis_recognition" target="_blank" rel="noreferrer">GitHub</a>
        <a href="https://www.linkedin.com/in/aleksey-tep/" target="_blank" rel="noreferrer">LinkedIn</a>
    </div>
    """,
    unsafe_allow_html=True
)

st.title("Tuberculosis recognition")
st.text("Choose an image and run recognize")


imageLocation = st.empty()

choosed_way = st.sidebar.selectbox(
    "Upload image or choose a preset",
    ("Not choosen", "Upload image", "Choose a preset")
)

if choosed_way == "Upload image":
    uploaded_file = st.sidebar.file_uploader("Upload an image")
    if uploaded_file is not None:
        _, ext = os.path.splitext(uploaded_file.name)
        ext = ext.lower()
        if ext in extensions:
            bytes_data = uploaded_file.getvalue()
            imageLocation.image(bytes_data, width=500)
            recognize = st.sidebar.button("Recognize!")
            if recognize:
                image = recognizer(bytes_data)
                imageLocation.image(image, width=500)
        else:
            st.error("Wrong extension! Please upload an image (jpg, jpeg, png, bmp, gif, tif, jp2, pcx, ppm, tga)")

elif choosed_way == "Choose a preset":
    selected_image = st.sidebar.selectbox(
        "Choose a preset image",
        ("Not choosen", "Normal 1", "Normal 2", "Normal 3", "Tuberculosis 1", "Tuberculosis 2", "Tuberculosis 3")
    )
    if selected_image != "Not choosen":
        image_path = path_dict[selected_image]
        with open(image_path, "rb") as file:
            bytes_data = file.read()
        imageLocation.image(bytes_data, width=500)
        recognize = st.sidebar.button("Recognize!")
        if recognize:
            image = recognizer(bytes_data)
            imageLocation.image(image, width=500)
