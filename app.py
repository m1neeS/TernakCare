import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import json
import os

# Page config
st.set_page_config(
    page_title="TernakCare - Deteksi Penyakit Ternak",
    page_icon="🐄",
    layout="centered"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 1rem;
    }
    .prediction-box {
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .healthy {
        background-color: #d4edda;
        border: 2px solid #28a745;
    }
    .disease {
        background-color: #f8d7da;
        border: 2px solid #dc3545;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("<h1 class='main-header'>🐄 TernakCare</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center;'>Sistem Deteksi Penyakit Ternak</h3>", unsafe_allow_html=True)
st.markdown("---")

# Class names dan info penyakit
CLASS_NAMES = ['foot-and-mouth', 'healthy', 'lumpy']
DISEASE_INFO = {
    'healthy': {
        'name': 'Sehat',
        'description': 'Ternak dalam kondisi sehat, tidak terdeteksi penyakit.',
        'color': 'green'
    },
    'lumpy': {
        'name': 'Lumpy Skin Disease (LSD)',
        'description': 'Penyakit kulit menular yang ditandai dengan benjolan/nodul pada kulit sapi.',
        'color': 'red'
    },
    'foot-and-mouth': {
        'name': 'Foot and Mouth Disease (FMD)',
        'description': 'Penyakit mulut dan kuku yang sangat menular, ditandai lesi di mulut dan kaki.',
        'color': 'red'
    }
}

@st.cache_resource
def load_model():
    """Load trained model"""
    model_path = 'ternakcare_best_model.keras'
    if os.path.exists(model_path):
        return tf.keras.models.load_model(model_path)
    else:
        st.error(f"Model tidak ditemukan: {model_path}")
        st.info("Pastikan sudah menjalankan training terlebih dahulu.")
        return None

def preprocess_image(image, target_size=(224, 224)):
    """Preprocess image for prediction"""
    img = image.resize(target_size)
    img_array = np.array(img)
    
    # Handle grayscale images
    if len(img_array.shape) == 2:
        img_array = np.stack([img_array] * 3, axis=-1)
    elif img_array.shape[-1] == 4:  # RGBA
        img_array = img_array[:, :, :3]
    
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def predict(model, image):
    """Make prediction on image"""
    processed_img = preprocess_image(image)
    predictions = model.predict(processed_img, verbose=0)
    
    predicted_idx = np.argmax(predictions[0])
    predicted_class = CLASS_NAMES[predicted_idx]
    confidence = predictions[0][predicted_idx] * 100
    
    all_probs = {CLASS_NAMES[i]: predictions[0][i] * 100 for i in range(len(CLASS_NAMES))}
    
    return predicted_class, confidence, all_probs

# Load model
model = load_model()

# File uploader
st.subheader("Upload Gambar Ternak")
uploaded_file = st.file_uploader(
    "Pilih gambar sapi untuk dianalisis",
    type=['jpg', 'jpeg', 'png'],
    help="Upload gambar sapi untuk mendeteksi penyakit"
)

# Sample images option
st.markdown("---")
use_sample = st.checkbox("Atau gunakan gambar sampel dari dataset")

if use_sample:
    sample_class = st.selectbox("Pilih kategori sampel:", CLASS_NAMES)
    sample_dir = f"data/{sample_class}"
    
    if os.path.exists(sample_dir):
        sample_images = [f for f in os.listdir(sample_dir) 
                        if f.lower().endswith(('.jpg', '.jpeg', '.png'))][:5]
        if sample_images:
            selected_sample = st.selectbox("Pilih gambar:", sample_images)
            sample_path = os.path.join(sample_dir, selected_sample)
            uploaded_file = sample_path

# Prediction
if uploaded_file is not None and model is not None:
    # Load image
    if isinstance(uploaded_file, str):  # Sample image path
        image = Image.open(uploaded_file)
    else:  # Uploaded file
        image = Image.open(uploaded_file)
    
    # Display image
    col1, col2 = st.columns(2)
    
    with col1:
        st.image(image, caption="Gambar Input", width="stretch")
    
    # Make prediction
    with st.spinner("Menganalisis gambar..."):
        predicted_class, confidence, all_probs = predict(model, image)
    
    # Display results
    with col2:
        disease_info = DISEASE_INFO[predicted_class]
        
        if predicted_class == 'healthy':
            st.success(f"**Hasil: {disease_info['name']}**")
        else:
            st.error(f"**Hasil: {disease_info['name']}**")
        
        st.metric("Confidence", f"{confidence:.1f}%")
        st.write(disease_info['description'])
    
    # Probability chart
    st.markdown("---")
    st.subheader("Detail Probabilitas")
    
    for class_name, prob in sorted(all_probs.items(), key=lambda x: x[1], reverse=True):
        info = DISEASE_INFO[class_name]
        st.write(f"**{info['name']}**")
        st.progress(float(prob) / 100)
        st.write(f"{prob:.2f}%")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    <p>TernakCare - Sistem Deteksi Penyakit Ternak berbasis AI</p>
    <p>Menggunakan MobileNetV2 Transfer Learning</p>
</div>
""", unsafe_allow_html=True)
