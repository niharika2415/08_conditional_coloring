import streamlit as st
import numpy as np
import cv2
from streamlit_drawable_canvas import st_canvas
import matplotlib.pyplot as plt

st.title("Quick Conditional Image Colorization (Overlay Hack)")

# Upload grayscale image
uploaded_file = st.file_uploader("Upload a grayscale image", type=["png","jpg","jpeg"])
if uploaded_file:
    # Read image
    file_bytes = np.frombuffer(uploaded_file.read(), np.uint8)
    gray_img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)

    # Resize to 256x256 for better GUI experience
    gray_img = cv2.resize(gray_img, (256, 256))
    color_img = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2RGB)

    # Pick color
    user_color = st.color_picker("Pick a color for selected region", "#FF0000")
    user_color_rgb = [int(user_color[i:i+2],16) for i in (1,3,5)]

    # Create a canvas to draw rectangle
    canvas_result = st_canvas(
        fill_color=None,
        stroke_width=5,
        stroke_color=user_color,
        background_image=gray_img,
        update_streamlit=True,
        height=256,
        width=256,
        drawing_mode="rect",
        key="canvas",
    )

    # Overlay rectangle color if drawn
    if canvas_result.json_data is not None:
        objects = canvas_result.json_data["objects"]
        for obj in objects:
            left = int(obj["left"])
            top = int(obj["top"])
            width = int(obj["width"])
            height = int(obj["height"])
            color_img[top:top+height, left:left+width] = user_color_rgb

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
