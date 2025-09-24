import streamlit as st
import numpy as np
import cv2
import matplotlib.pyplot as plt
from streamlit_drawable_canvas import st_canvas

st.title("Quick Conditional Image Colorization (Overlay Hack)")

# Upload grayscale image
uploaded_file = st.file_uploader("Upload a grayscale image", type=["png", "jpg", "jpeg"])
if uploaded_file:
    # Read and resize image
    file_bytes = np.frombuffer(uploaded_file.read(), np.uint8)
    gray_img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
    gray_img = cv2.resize(gray_img, (256, 256))

    st.image(gray_img, caption="Uploaded Grayscale", use_container_width=True)

    # Convert to 3-channel RGB for coloring
    color_img = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2RGB)

    # Pick color
    user_color = st.color_picker("Pick a color to draw", "#FF0000")

    # Canvas for drawing
    canvas_result = st_canvas(
        fill_color=None,
        stroke_width=15,
        stroke_color=user_color,
        background_color=None,
        background_image=color_img,
        update_streamlit=True,
        height=256,
        width=256,
        drawing_mode="rect",
        key="canvas",
    )

    # Apply overlay to image if any rectangles drawn
    if canvas_result.json_data is not None:
        shapes = canvas_result.json_data["objects"]
        for shape in shapes:
            left = int(shape["left"])
            top = int(shape["top"])
            width = int(shape["width"])
            height = int(shape["height"])
            color_img[top:top+height, left:left+width] = [int(user_color[i:i+2], 16) for i in (1,3,5)]

    # Show side-by-side
    st.subheader("Result")
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(gray_img, cmap="gray")
    axes[0].set_title("Grayscale Input")
    axes[0].axis("off")

    axes[1].imshow(color_img)
    axes[1].set_title("Overlayed Output")
    axes[1].axis("off")

    st.pyplot(fig)
