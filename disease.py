from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
import logging
import os
from keras.layers import TFSMLayer
from dotenv import load_dotenv


# Load environment variables from .env
load_dotenv()

# Initialize FastAPI app
app = FastAPI()

# CORS middleware for allowing specific origins (read from env or fallback)
origins = os.getenv("CORS_ORIGINS", "http://localhost,http://127.0.0.1:5500").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the pre-trained model path and serving endpoint from env variables
MODEL_PATH = os.getenv("MODEL_PATH", "model/7")
CALL_ENDPOINT = os.getenv("CALL_ENDPOINT", "serving_default")

MODEL = TFSMLayer(MODEL_PATH, call_endpoint=CALL_ENDPOINT)

CLASS_NAMES = os.getenv("CLASS_NAMES", "Early Blight,Late Blight,Healthy").split(",")

# Logger setup
logging.basicConfig(level=logging.INFO)

# Health check endpoint
@app.get("/ping")
async def ping():
    return {"message": "Hello, I am alive"}

# Function to read and preprocess the image file
def read_file_as_image(data) -> np.ndarray:
    try:
        image = Image.open(BytesIO(data)).convert("RGB")  # Force 3 channels
        image = np.array(image)
        image = tf.image.resize(image, (256, 256))
        image = image / 255.0
        return image
    except Exception as e:
        logging.error(f"Error reading image: {e}")
        raise HTTPException(status_code=400, detail="Failed to read image file")

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        image = read_file_as_image(await file.read())
        img_batch = np.expand_dims(image, 0)
        # Run prediction
        predictions = MODEL(img_batch, training=False)
        logging.info(f"Raw predictions: {predictions}")

        # Handle dict output
        if isinstance(predictions, dict):
            predictions = list(predictions.values())[0]

        # Convert tensor to numpy if needed
        if hasattr(predictions, "numpy"):
            predictions = predictions.numpy()

        predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
        confidence = np.max(predictions[0])

        return {
            "class": predicted_class,
            "confidence": float(confidence) * 100
        }

    except Exception as e:
        logging.error(f"Error during prediction: {e}")
        raise HTTPException(status_code=500, detail="Failed to make prediction")

# Note: No need for UVicorn run block for deployment to Render
