import streamlit as st
import numpy as np
import cv2
import matplotlib.pyplot as plt

st.title("Quick Conditional Image Colorization (Overlay Hack)")

# Upload grayscale image
uploaded_file = st.file_uploader("Upload a grayscale image", type=["png","jpg","jpeg"])
if uploaded_file:
    # Read image
    file_bytes = np.frombuffer(uploaded_file.read(), np.uint8)
    gray_img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
    gray_img = cv2.resize(gray_img, (256,256))  # Resize to bigger size for clarity

    st.image(gray_img, caption="Uploaded Grayscale", use_container_width=True)

    # User selects rectangle region
    st.write("Select region to color (enter pixel coordinates)")
    x1 = st.number_input("x1", 0, gray_img.shape[1]-1, 0)
    y1 = st.number_input("y1", 0, gray_img.shape[0]-1, 0)
    x2 = st.number_input("x2", 0, gray_img.shape[1]-1, gray_img.shape[1]-1)
    y2 = st.number_input("y2", 0, gray_img.shape[0]-1, gray_img.shape[0]-1)

    # Pick color
    user_color = st.color_picker("Pick a color for selected region", "#FF0000")
    # Convert HEX to RGB (0-255)
    user_color_rgb = [int(user_color[i:i+2],16) for i in (1,3,5)]

    # Convert grayscale to 3-channel image
    color_img = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2RGB)

    # Overlay user color on selected region
    color_img[y1:y2, x1:x2] = user_color_rgb

    # Show side-by-side
    st.subheader("Result")
    fig, axes = plt.subplots(1,2, figsize=(10,5))
    axes[0].imshow(gray_img, cmap='gray')
    axes[0].set_title("Grayscale Input")
    axes[0].axis('off')

    axes[1].imshow(color_img)
    axes[1].set_title("Overlayed Output")
    axes[1].axis('off')

    st.pyplot(fig)
