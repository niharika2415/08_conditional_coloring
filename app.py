import streamlit as st
import numpy as np
import cv2
from PIL import Image
from streamlit_drawable_canvas import st_canvas
import matplotlib.pyplot as plt

st.title("Quick Conditional Image Colorization (Rectangle Fill)")

# Upload grayscale image
uploaded_file = st.file_uploader("Upload a grayscale image", type=["png", "jpg", "jpeg"])

if uploaded_file:
    # Read image
    file_bytes = np.frombuffer(uploaded_file.read(), np.uint8)
    gray_img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
    gray_img = cv2.resize(gray_img, (256, 256))  # Resize for clarity

    # Convert grayscale to 3-channel for coloring
    color_img = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2RGB)
    pil_color_img = Image.fromarray(color_img)

    # Show uploaded grayscale image
    st.image(pil_color_img, caption="Uploaded Grayscale", use_container_width=True)

    # Pick color
    user_color = st.color_picker("Pick a color for rectangles", "#FF0000")
    # Convert HEX to RGB
    user_color_rgb = tuple(int(user_color[i:i+2], 16) for i in (1, 3, 5))

    # Create drawable canvas
    canvas_result = st_canvas(
        fill_color=user_color,      # Fill rectangles with selected color
        stroke_width=0,             # No extra stroke
        background_image=pil_color_img,
        update_streamlit=True,
        height=256,
        width=256,
        drawing_mode="rect",
        key="canvas"
    )

    # Overlay rectangles on original image
    if canvas_result.json_data is not None:
        final_img = color_img.copy()
        for obj in canvas_result.json_data["objects"]:
            if obj["type"] == "rect":
                x1, y1 = int(obj["left"]), int(obj["top"])
                x2, y2 = int(obj["left"] + obj["width"]), int(obj["top"] + obj["height"])
                final_img[y1:y2, x1:x2] = user_color_rgb

        # Show result
        st.subheader("Colorized Output")
        fig, ax = plt.subplots(figsize=(5,5))
        ax.imshow(final_img)
        ax.axis('off')
        st.pyplot(fig)
