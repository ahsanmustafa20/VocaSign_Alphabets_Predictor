# Save this as app.py and run with: streamlit run app.py

import streamlit as st
from PIL import Image
import torch
from torchvision import transforms, models
import torch.nn.functional as F
import numpy as np
from gtts import gTTS
import tempfile
import os

# ========================
# 1. Config
# ========================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_CLASSES = 29  # total classes in your dataset
MODEL_PATH = "alphabet_classifier.pth"

# Load class names
from torchvision import datasets
train_dataset = datasets.ImageFolder(r"D:\Courses\AKTI Gen AI\Project\EDA\Alphabets\alphabet_dataset\train")  # path to train folder
class_names = train_dataset.classes

# ========================
# 2. Load Model
# ========================
model = models.resnet18(pretrained=True)
model.fc = torch.nn.Linear(model.fc.in_features, NUM_CLASSES)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model = model.to(DEVICE)
model.eval()

# ========================
# 3. Image Transform
# ========================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ========================
# 4. Streamlit Interface
# ========================
# st.image("logo.png", width=100)
# st.title("🖐️ VocaSign-Alphabet Predictor")

st.set_page_config(page_title="VocaSign-Alphabet Predictor")


import base64

def img_to_bytes(img_path):
    with open(img_path, "rb") as f:
        return base64.b64encode(f.read()).decode()

logo = img_to_bytes("logo.png")
st.markdown(
    f"""
    <div style="display:flex; align-items:center;">
        <img src="data:image/png;base64,{logo}" width="150" style="margin-right:15px;">
        <h1 style="margin:0;">VocaSign-Alphabet Predictor</h1>
    </div>
    """,
    unsafe_allow_html=True
)

image = None

option = st.radio("Choose input method:", ("Upload Image", "Capture Image"))

if option == "Upload Image":
    uploaded_file = st.file_uploader("Upload an image", type=['jpg', 'png', 'jpeg'])
    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")

elif option == "Capture Image":
    captured_file = st.camera_input("Capture an image")
    if captured_file:
        image = Image.open(captured_file).convert("RGB")

# ========================
# 5. Predict Button
# ========================
if image is not None:
    st.image(image, caption="Input Image", use_container_width=True)

    if st.button("Predict Letter"):
        # Transform image
        img = transform(image).unsqueeze(0).to(DEVICE)
        
        # Prediction
        with torch.no_grad():
            outputs = model(img)
            probs = F.softmax(outputs, dim=1)
            top5_prob, top5_idx = torch.topk(probs, 5)
            
            top5_prob = top5_prob.cpu().numpy()[0]
            top5_idx = top5_idx.cpu().numpy()[0]
            top5_letters = [class_names[i] for i in top5_idx]
        
        # Show top-1 prediction
        st.success(f"Predicted Letter: {top5_letters[0]}")

        # Show top-5 probabilities
        st.write("### Top-5 Probabilities")
        for letter, prob in zip(top5_letters, top5_prob):
            st.write(f"{letter}: {prob*100:.2f}%")
        
        # Generate audio for top-1 letter
        tts = gTTS(top5_letters[0])
        tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        tts.save(tmp_file.name)
        st.audio(tmp_file.name)
        # Optional: remove temp file after audio
        # os.unlink(tmp_file.name)
