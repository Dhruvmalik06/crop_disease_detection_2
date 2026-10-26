from unittest import result
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
from pathlib import Path




app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost",
        "http://localhost:3000",
        "https://crop-disease-detection-2-1.onrender.com",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Locate the repository root using this file's location.
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "1"

print("Looking for model at:", MODEL_PATH)
print("Model directory exists:", MODEL_PATH.is_dir())

if not (MODEL_PATH / "saved_model.pb").is_file():
    raise FileNotFoundError(
        f"SavedModel not found at {MODEL_PATH}. "
        "Check the model folder and Render Root Directory."
    )

# Load the model exported with model.export()
MODEL = tf.saved_model.load(str(MODEL_PATH))

if "serving_default" not in MODEL.signatures:
    raise RuntimeError(
        f"serving_default signature not found. "
        f"Available signatures: {list(MODEL.signatures.keys())}"
    )

infer = MODEL.signatures["serving_default"]

CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]

print("Model loaded successfully!")
print("Available signatures:", list(MODEL.signatures.keys()))
print("Input details:", infer.structured_input_signature)
print("Output details:", infer.structured_outputs)


@app.get("/")
def home():
    return {"message": "Crop Disease Detection API is running!"}


def read_file_as_image(data) -> np.ndarray:
    image = Image.open(BytesIO(data)).convert("RGB")
    return np.array(image)


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image = read_file_as_image(await file.read())
    image_batch = np.expand_dims(image, axis=0)

    input_name = list(
        infer.structured_input_signature[1].keys()
    )[0]

    input_spec = infer.structured_input_signature[1][input_name]

    image_tensor = tf.convert_to_tensor(
        image_batch,
        dtype=input_spec.dtype,
    )

    outputs = infer(**{input_name: image_tensor})
    output_tensor = next(iter(outputs.values()))
    predictions = output_tensor.numpy()[0]

    predicted_index = int(np.argmax(predictions))

    if predicted_index >= len(CLASS_NAMES):
        raise RuntimeError(
            f"Model returned unexpected output shape: {predictions.shape}"
        )

    confidence = float(predictions[predicted_index])
    predicted_class = CLASS_NAMES[predicted_index]

    return {
        "class": predicted_class,
        "predicted_class_index": predicted_index,
        "confidence": confidence,
        "prediction_scores": predictions.tolist(),
    }



    
if __name__ == "__main__":
    import os
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 8000))
    )    
    

