from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
import logging

# Initialize FastAPI app
app = FastAPI()

# CORS middleware for allowing specific origins
origins = ["http://localhost", "http://127.0.0.1:5500"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the pre-trained model from the correct path
from keras.layers import TFSMLayer

MODEL = TFSMLayer("model/7", call_endpoint="serving_default")

CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]

# Logger setup
logging.basicConfig(level=logging.INFO)

# Health check endpoint
@app.get("/ping")
async def ping():
    return {"message": "Hello, I am alive"}

# Function to read and preprocess the image file
def read_file_as_image(data) -> np.ndarray:
    try:
        image = Image.open(BytesIO(data)).convert("RGB")  # 🔹 Force 3 channels
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

        # Log the raw prediction for debugging
        logging.info(f"Raw predictions: {predictions}")

        # Some models output dicts instead of tensors
        if isinstance(predictions, dict):
            predictions = list(predictions.values())[0]

        # Convert to numpy if it’s still a tensor
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


# Run the app using Uvicorn
# if __name__ == "__main__":
#     uvicorn.run(app, host='localhost', port=8000)