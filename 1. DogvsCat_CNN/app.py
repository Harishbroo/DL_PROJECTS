import streamlit as st
from tensorflow.keras.models import load_model  # type: ignore
import cv2
import numpy as np


# Load model only when needed
@st.cache_resource(show_spinner="Loading model...")
def get_model():
    return load_model("dog_cat_model.keras", compile=False)


# Streamlit UI
st.title("🐶🐱 Dog vs Cat Classifier")

st.write("Upload an image of a dog or cat")


# Upload image
uploaded_file = st.file_uploader(
    "Choose an image...",
    type=["jpg", "jpeg", "png"]
)


if uploaded_file is not None:

    # Load model
    model = get_model()

    # Read image
    file_bytes = np.asarray(
        bytearray(uploaded_file.read()),
        dtype=np.uint8
    )

    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    # Display uploaded image
    st.image(
        image,
        channels="BGR",
        caption="Uploaded Image"
    )

    # Resize image
    image_resized = cv2.resize(
        image,
        (256, 256)
    )

    # Convert BGR to RGB
    image_rgb = cv2.cvtColor(
        image_resized,
        cv2.COLOR_BGR2RGB
    )

    # Normalize pixel values
    image_normalized = image_rgb / 255.0

    # Add batch dimension
    image_input = np.expand_dims(
        image_normalized,
        axis=0
    )

    # Make prediction
    prediction = model.predict(image_input)

    # Display result
    if prediction[0][0] > 0.5:
        st.success("🐶 Prediction: Dog")
    else:
        st.success("🐱 Prediction: Cat")