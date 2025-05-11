import streamlit as st
import torch
from PIL import Image
from model import ImageEncoder
from ultis import predict_image_v2  # replace with your actual module
import os
import gdown

torch.classes.__path__ = [os.path.join(torch.__path__[0], torch.classes.__file__)] 

device = "cpu"
MODEL_PATH = "image_encoder_leaf_clip.bin"
GOOGLE_DRIVE_URL = "https://drive.google.com/file/d/1wvZGP_GZzt0O2Q9HfEpQAG0vC1f0OjwA/view?usp=sharing"


@st.cache_resource
def load_model():
    model = ImageEncoder()
    model = model.to(device)
    model.load_state_dict(torch.load("image_encoder_leaf_clip.bin", weights_only=True, map_location="cpu"))
    model.eval()
    return model

# Page config
st.set_page_config(page_title="Leaf Disease Prediction", layout="wide")
st.title("🌿 Leaf Disease Image-to-Text Classification")

# Upload section
meta_path = "class_description.json"
uploaded_file = st.file_uploader("Upload a leaf image", type=["jpg", "jpeg", "png","JPG"])
top_k = st.slider("Select number of top predictions (K)", min_value=1, max_value=20, value=10)

# Layout: 2 Columns
left_col, right_col = st.columns(2)

with left_col:
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.success("✅ Image uploaded successfully!")
        st.image(image, caption="Uploaded Leaf Image", use_container_width=True)

        with right_col:
            predict_clicked = st.button("🔍 Predict")

            if predict_clicked and uploaded_file is not None:
                with st.spinner("Analyzing... Please wait."):
                    model = load_model()
                    predictions = predict_image_v2(model, uploaded_file, meta_path, top_k=top_k)

                st.markdown("### 📊 Top Predictions")

                for i, (text, prob) in enumerate(predictions):
                    confidence = prob * 100

                    st.markdown(f"""
                        <div style="padding:10px 15px;color:black;margin-bottom: 10px; border: 1px solid #ddd; border-radius: 8px; background-color: #f9f9f9;">
                            <strong>{i+1}. {text}</strong> — <span style="color: #4CAF50;"><strong>{confidence:.2f}%</strong></span>
                        </div>
                    """, unsafe_allow_html=True)

                    st.progress(prob)
