import streamlit as st
import numpy as np
import cv2
from PIL import Image
from streamlit_drawable_canvas import st_canvas
import matplotlib.pyplot as plt

st.title("Conditional Image Colorization")

# Upload grayscale image
uploaded_file = st.file_uploader("Upload a grayscale image", type=["png", "jpg", "jpeg"])

if uploaded_file:
    # Read image
    file_bytes = np.frombuffer(uploaded_file.read(), np.uint8)
    gray_img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
    gray_img = cv2.resize(gray_img, (256, 256))  # Resize for better visualization

    # Convert grayscale to 3-channel for color overlay
    color_img = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2RGB)
    pil_color_img = Image.fromarray(color_img)

    # Pick color
    user_color = st.color_picker("Pick a color for rectangles", "#FF0000")

    # Create drawable canvas
    canvas_result = st_canvas(
        fill_color=user_color,        # Auto-fill rectangles
        stroke_width=0,               # No extra stroke
        stroke_color=user_color,      # Keep stroke same as fill
        background_image=pil_color_img,
        update_streamlit=True,
        height=256,
        width=256,
        drawing_mode="rect",          # Only rectangles
        key="canvas"
    )

    # Display the updated image
    if canvas_result.image_data is not None:
        st.subheader("Colorized Output")
        fig, ax = plt.subplots(figsize=(5,5))
        ax.imshow(canvas_result.image_data.astype(np.uint8))
        ax.axis('off')
        st.pyplot(fig)
