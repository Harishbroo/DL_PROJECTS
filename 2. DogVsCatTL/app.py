import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# Page configuration
st.set_page_config(
    page_title="Cat vs Dog - Transfer Learning",
    page_icon="🐱",
    layout="centered"
)

# Load the trained model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("transfer_learning_dog_cat.keras")

model = load_model()

# App interface
st.title("🐱 Cat vs Dog Classifier")
st.write("Transfer Learning with MobileNetV2")

uploaded_file = st.file_uploader(
    "Upload a cat or dog image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    st.image(image, caption="Uploaded Image", use_container_width=True)

    if st.button("Predict"):
        # Resize image to the model's expected input size
        image = image.resize((224, 224))

        # Convert image to a NumPy array
        image_array = np.array(image, dtype=np.float32)

        # Apply MobileNetV2 preprocessing
        image_array = preprocess_input(image_array)

        # Add batch dimension: (224, 224, 3) -> (1, 224, 224, 3)
        image_array = np.expand_dims(image_array, axis=0)

        # Make prediction
        prediction = float(model.predict(image_array, verbose=0)[0][0])

        # Our training labels: 0 = Cat, 1 = Dog
        if prediction >= 0.5:
            st.success("Prediction: Dog 🐶")
            st.write(f"Dog score: {prediction * 100:.2f}%")
        else:
            st.success("Prediction: Cat 🐱")
            st.write(f"Cat score: {(1 - prediction) * 100:.2f}%")