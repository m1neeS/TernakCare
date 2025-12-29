from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import os

app = FastAPI(
    title="TernakCare API",
    description="API untuk deteksi penyakit ternak menggunakan AI",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Class names dan info penyakit
CLASS_NAMES = ['foot-and-mouth', 'healthy', 'lumpy']
DISEASE_INFO = {
    'healthy': {
        'name': 'Sehat',
        'description': 'Ternak dalam kondisi sehat, tidak terdeteksi penyakit.',
        'recommendation': 'Lanjutkan perawatan rutin dan jaga kebersihan kandang.'
    },
    'lumpy': {
        'name': 'Lumpy Skin Disease (LSD)',
        'description': 'Penyakit kulit menular yang ditandai dengan benjolan/nodul pada kulit sapi.',
        'recommendation': 'Segera isolasi ternak dan hubungi dokter hewan. Lakukan vaksinasi pada ternak sehat.'
    },
    'foot-and-mouth': {
        'name': 'Foot and Mouth Disease (FMD)',
        'description': 'Penyakit mulut dan kuku yang sangat menular, ditandai lesi di mulut dan kaki.',
        'recommendation': 'Isolasi segera, laporkan ke dinas peternakan, dan lakukan desinfeksi kandang.'
    }
}

# Load model
model = None

def load_model():
    global model
    model_path = 'ternakcare_best_model.keras'
    if os.path.exists(model_path):
        model = tf.keras.models.load_model(model_path)
        print("✅ Model loaded successfully")
    else:
        print(f"❌ Model not found: {model_path}")

@app.on_event("startup")
async def startup_event():
    load_model()

def preprocess_image(image: Image.Image, target_size=(224, 224)):
    """Preprocess image for prediction"""
    img = image.resize(target_size)
    img_array = np.array(img)
    
    if len(img_array.shape) == 2:
        img_array = np.stack([img_array] * 3, axis=-1)
    elif img_array.shape[-1] == 4:
        img_array = img_array[:, :, :3]
    
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

@app.get("/")
async def root():
    return {
        "message": "TernakCare API",
        "version": "1.0.0",
        "endpoints": {
            "predict": "/predict",
            "health": "/health",
            "classes": "/classes"
        }
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None
    }

@app.get("/classes")
async def get_classes():
    return {
        "classes": CLASS_NAMES,
        "details": DISEASE_INFO
    }

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Validate file type
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # Read and process image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        
        # Preprocess and predict
        processed_img = preprocess_image(image)
        predictions = model.predict(processed_img, verbose=0)
        
        predicted_idx = int(np.argmax(predictions[0]))
        predicted_class = CLASS_NAMES[predicted_idx]
        confidence = float(predictions[0][predicted_idx]) * 100
        
        # All probabilities
        probabilities = {
            CLASS_NAMES[i]: float(predictions[0][i]) * 100 
            for i in range(len(CLASS_NAMES))
        }
        
        disease_info = DISEASE_INFO[predicted_class]
        
        return {
            "success": True,
            "prediction": {
                "class": predicted_class,
                "name": disease_info['name'],
                "confidence": round(confidence, 2),
                "description": disease_info['description'],
                "recommendation": disease_info['recommendation']
            },
            "probabilities": {
                k: round(v, 2) for k, v in probabilities.items()
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
