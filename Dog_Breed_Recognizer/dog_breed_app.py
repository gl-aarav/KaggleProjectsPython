import cv2
from keras.models import load_model
import numpy as np
import streamlit as st

# Loading the model
model = load_model('dog_breed_model.keras')

CLASS_NAMES = ['scottish_deerhound', 'maltese_dog', 'bernese_mountain_dog']

# App title
st.title("Dog Breed Predictor")
st.markdown("Upload an image of a dog, and the application will predict its breed.")

# File uploader
dog_image = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

# Predict button
submit = st.button("Predict")

if submit:
    if dog_image is not None:
        # Read image bytes and decode
        file_bytes = np.asarray(bytearray(dog_image.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        # Show uploaded image
        st.image(img, channels="BGR")
        
        # Resize and normalize
        img_resized = cv2.resize(img, (224, 224))
        img_resized = img_resized / 255.0  # normalize
        img_resized = np.expand_dims(img_resized, axis=0)  # shape: (1, 224, 224, 3)
        
        # Predict
        Y_pred = model.predict(img_resized)
        
        # Show result
        st.title(f"Predicted Breed: {CLASS_NAMES[np.argmax(Y_pred)]}")
    else:
        st.warning("Please upload an image before predicting.")
