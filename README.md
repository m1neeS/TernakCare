# 🐄 TernakCare - Livestock Disease Detection

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://tensorflow.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![CNN](https://img.shields.io/badge/Model-MobileNetV2-purple.svg)](https://arxiv.org/abs/1801.04381)

**TernakCare** adalah sistem deteksi penyakit ternak berbasis Deep Learning yang menggunakan arsitektur **MobileNetV2** untuk mengklasifikasikan kondisi kesehatan sapi dari gambar.

![TernakCare Banner](https://via.placeholder.com/800x200/4ECDC4/FFFFFF?text=TernakCare+-+Livestock+Disease+Detection)

## 🎯 Features

- ✅ Deteksi **Lumpy Skin Disease (LSD)**
- ✅ Deteksi **Foot and Mouth Disease (FMD)**
- ✅ Identifikasi sapi **sehat**
- ✅ Transfer Learning dengan MobileNetV2
- ✅ Export ke TFLite untuk mobile deployment
- ✅ High accuracy classification

## 📊 Dataset

| Class | Samples | Description |
|-------|---------|-------------|
| Healthy | ~900+ | Sapi sehat dari berbagai ras |
| Lumpy Skin Disease | ~1000+ | Lesi kulit berbentuk benjolan |
| Foot and Mouth Disease | ~600+ | Lesi di mulut, lidah, dan kaki |

## 🏗️ Model Architecture

```
MobileNetV2 (pretrained ImageNet)
    │
    ├── GlobalAveragePooling2D
    ├── BatchNormalization
    ├── Dropout(0.5)
    ├── Dense(256, ReLU)
    ├── BatchNormalization
    ├── Dropout(0.3)
    ├── Dense(128, ReLU)
    ├── Dropout(0.2)
    └── Dense(3, Softmax)
```

## 🚀 Quick Start

### Prerequisites

```bash
pip install tensorflow matplotlib seaborn scikit-learn pillow
```

### Training

1. Clone repository
```bash
git clone https://github.com/yourusername/ternakcare.git
cd ternakcare
```

2. Prepare dataset structure
```
data/
├── healthy/
├── lumpy/
└── foot-and-mouth/
```

3. Run notebook
```bash
jupyter notebook TernakCare_Disease_Detection.ipynb
```

### Inference

```python
from tensorflow import keras
import numpy as np
from tensorflow.keras.preprocessing import image

# Load model
model = keras.models.load_model('ternakcare_best_model.keras')
class_names = ['foot-and-mouth', 'healthy', 'lumpy']

# Predict
img = image.load_img('path/to/image.jpg', target_size=(224, 224))
img_array = image.img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

predictions = model.predict(img_array)
predicted_class = class_names[np.argmax(predictions[0])]
confidence = np.max(predictions[0]) * 100

print(f"Prediction: {predicted_class} ({confidence:.2f}%)")
```

## 📁 Project Structure

```
ternakcare/
├── data/
│   ├── healthy/
│   ├── lumpy/
│   └── foot-and-mouth/
├── TernakCare_Disease_Detection.ipynb
├── README.md
├── requirements.txt
├── ternakcare_best_model.keras
├── ternakcare_model.tflite
├── class_indices.json
└── model_config.json
```

## 📈 Results

| Metric | Value |
|--------|-------|
| Training Accuracy | ~95%+ |
| Validation Accuracy | ~90%+ |
| Model Size (TFLite) | ~10 MB |

## 🛠️ Technologies

- **Python 3.8+**
- **TensorFlow 2.x**
- **MobileNetV2** (Transfer Learning)
- **Keras** (High-level API)
- **Matplotlib & Seaborn** (Visualization)
- **Scikit-learn** (Metrics)

## 📱 Mobile Deployment

Model tersedia dalam format TFLite untuk deployment di Android/iOS:

```python
# Convert to TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👨‍💻 Author

**TernakCare Team**

---

⭐ Star this repo if you find it helpful!