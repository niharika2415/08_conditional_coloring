import streamlit as st
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from streamlit_drawable_canvas import st_canvas

st.title("Quick Conditional Image Colorization (Overlay Hack)")

# Upload grayscale image
uploaded_file = st.file_uploader("Upload a grayscale image", type=["png","jpg","jpeg"])
if uploaded_file:
    # Read and resize image
    file_bytes = np.frombuffer(uploaded_file.read(), np.uint8)
    gray_img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
    gray_img = cv2.resize(gray_img, (256, 256))

    # Display original grayscale
    st.image(gray_img, caption="Uploaded Grayscale", use_container_width=True)

    # Pick color
    user_color = st.color_picker("Pick a color for selected region", "#FF0000")

    # Convert grayscale to 3-channel RGB
    color_img = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2RGB)

    # Convert to PIL Image for canvas background
    pil_bg = Image.fromarray(color_img)

    # Canvas for drawing rectangles
    canvas_result = st_canvas(
        fill_color=None,
        stroke_width=15,
        stroke_color=user_color,  # hex string
        background_color=None,
        background_image=pil_bg,
        update_streamlit=True,
        height=256,
        width=256,
        drawing_mode="rect",
        key="canvas",
    )

    # Overlay the rectangles drawn on the canvas
    if canvas_result.json_data is not None:
        for obj in canvas_result.json_data["objects"]:
            if obj["type"] == "rect":
                # Extract rectangle coordinates
                left = int(obj["left"])
                top = int(obj["top"])
                width = int(obj["width"])
                height = int(obj["height"])
                color_img[top:top+height, left:left+width] = [int(user_color[i:i+2],16) for i in (1,3,5)]

    # Show side-by-side result
    st.subheader("Result")
    fig, axes = plt.subplots(1,2, figsize=(10,5))
    axes[0].imshow(gray_img, cmap='gray')
    axes[0].set_title("Grayscale Input")
    axes[0].axis('off')

    axes[1].imshow(color_img)
    axes[1].set_title("Overlayed Output")
    axes[1].axis('off')

    st.pyplot(fig)
