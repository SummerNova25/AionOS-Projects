import streamlit as st
from PIL import Image
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration

# Load model and processor once
@st.cache_resource(show_spinner=True)
def load_model():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    return processor, model

processor, model = load_model()

st.title("Image Caption Generator")

uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Prepare input for model
    inputs = processor(image, return_tensors="pt")

    # Generate caption
    out = model.generate(**inputs)
    caption = processor.decode(out[0], skip_special_tokens=True)

    st.markdown(f"**Caption:** {caption}")
