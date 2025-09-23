import streamlit as st
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import gdown
import os

st.title("Conditional Image Colorization")

# Download model from Google Drive
MODEL_FILE = "conditional_colorizer.h5"
FILE_ID = "1gGL7_YPoEoaVXCnJJ3sj-aV7iwjy5FSj"  
if not os.path.exists(MODEL_FILE):
    url = f"https://drive.google.com/uc?id={FILE_ID}"
    gdown.download(url, MODEL_FILE, quiet=False)

# Load model
model = load_model(MODEL_FILE)

# Upload grayscale image
uploaded_file = st.file_uploader("Upload a grayscale image", type=["png","jpg","jpeg"])
if uploaded_file:
    file_bytes = np.frombuffer(uploaded_file.read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (32,32))  # use same size as training
    grayscale = img[..., np.newaxis] / 255.0  # normalize
    st.image(img, caption="Uploaded Grayscale", use_column_width=True)

    # User selects rectangle region
    st.write("Select region to color (enter pixel coordinates)")
    x1 = st.number_input("x1", 0, 31, 0)
    y1 = st.number_input("y1", 0, 31, 0)
    x2 = st.number_input("x2", 0, 31, 31)
    y2 = st.number_input("y2", 0, 31, 31)

    # User picks color
    user_color = st.color_picker("Pick a color for selected region", "#FF0000")
    # Convert HEX to normalized RGB
    user_color = np.array([int(user_color[i:i+2],16)/255.0 for i in (1,3,5)])

    # Prepare mask + hint 
    mask = np.zeros_like(grayscale)
    mask[y1:y2, x1:x2, 0] = 1.0
    hint = np.zeros((32,32,3))
    hint[y1:y2, x1:x2, :] = user_color

    input_with_hint = np.concatenate([grayscale, mask, hint], axis=-1)
    input_with_hint = np.expand_dims(input_with_hint, axis=0)

    # Predict 
    colorized = model.predict(input_with_hint)[0]

    # Show results
    st.subheader("Conditional Colorized Output")
    plt.figure(figsize=(8,4))
    plt.subplot(1,2,1)
    plt.title("Grayscale Input")
    plt.imshow(grayscale.squeeze(), cmap='gray')
    plt.axis("off")

    plt.subplot(1,2,2)
    plt.title("Colorized Output")
    plt.imshow(colorized)
    plt.axis("off")
    st.pyplot(plt)
