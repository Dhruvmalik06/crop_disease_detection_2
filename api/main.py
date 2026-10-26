from unittest import result
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf


app = FastAPI()


origins = [
    "http://localhost",
    "http://localhost:3000",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL = tf.saved_model.load(str("../crop_disease_detection_2/models/1"))
CLASS_NAMES = ['Early Blight', 'Late Blight', 'Healthy']
print("Available signatures:", list(MODEL.signatures.keys()))

infer = MODEL.signatures["serving_default"]

print("Input details:", infer.structured_input_signature)
print("Output details:", infer.structured_outputs)


@app.get("/")
def home():
    return {"message": "Crop Disease Detection API is running!"}

def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image



@app.post("/predict")
async def predict(file: UploadFile = File(...)):
      image = read_file_as_image(await file.read())
      image_batch = np.expand_dims(image, 0)
      print("Model type:", type(MODEL))
      print("Available signatures:", list(MODEL.signatures.keys()))

      infer = MODEL.signatures["serving_default"]

      print("Input signature:", infer.structured_input_signature)
      print("Output signature:", infer.structured_outputs)

      print("Image batch shape:", image_batch.shape)
      print("Image batch dtype:", image_batch.dtype)
      
    
      input_name = list(
          infer.structured_input_signature[1].keys()
      )[0]

      input_spec = infer.structured_input_signature[1][input_name]

   
      image_tensor = tf.convert_to_tensor(
          image_batch,
          dtype=input_spec.dtype
     )

    
      result = infer(**{input_name: image_tensor})

      output_tensor = next(iter(result.values()))
      predictions = output_tensor.numpy()[0]

      
      predicted_index = int(np.argmax(predictions))
      confidence = float(predictions[predicted_index])

      predicted_class = CLASS_NAMES[predicted_index]

      print("All prediction scores:", predictions)
      print("Predicted class index:", predicted_index)
      print("Predicted class:", predicted_class)
      print("Confidence score:", confidence)

      return {
          "class": predicted_class,
          "predicted_class_index": predicted_index,
          "confidence": confidence,
          "prediction_scores": predictions.tolist()
  }

    
    
    
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
