import streamlit as st
import numpy as np
from PIL import Image
from model import PreTrainedColorizationModel

#Main Streamlit App 
st.title("Interactive Image Colorization (Real-World Approach)")
st.markdown("Upload a grayscale image and use the color picker to add a hint.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read the image and convert it to grayscale
    original_img = Image.open(uploaded_file).convert("L")
    st.image(original_img, caption="Original Grayscale Image", use_column_width=True)

    st.sidebar.header("Add a Color Hint")
    hint_color_hex = st.sidebar.color_picker("Pick a color hint", "#1c99f0")
    
    st.sidebar.markdown(
        """
        ### How to use:
        1. Pick a color from the palette.
        2. Click the 'Apply' button.
        3. The pre-trained model will intelligently apply the color.
        """
    )
    st.sidebar.info("This project simulates a real-world deployment by using a pre-trained model.")

    if st.sidebar.button("Apply Color Hint"):
        with st.spinner("Applying colorization..."):
            # Load the pre-trained model
            model = PreTrainedColorizationModel()
            # Perform colorization using the model
            colorized_image = model.colorize(original_img, hint_color_hex)
            
            st.image(colorized_image, caption="Colorized Image", use_column_width=True)
            st.success("Colorization complete!")
