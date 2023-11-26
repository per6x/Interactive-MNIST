# app/main.py
import base64
import io
import json
from pathlib import Path
from typing import List

import numpy as np
from fastapi import (Body, FastAPI, File, HTTPException, UploadFile, WebSocket,
                     WebSocketDisconnect)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from tensorflow.keras.models import load_model

app = FastAPI()


# Mount the 'static' folder to serve static files (like the HTML page)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Load the Keras model
model = load_model('models/mnist')

# Define a function to preprocess the image
def preprocess_image(image):
    img = image.convert("L")
    img = img.resize((28, 28))
    img = np.array(img) / 255.0

    img_array = img.reshape((1, 28, 28, 1))
    return img_array

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            data_dict = json.loads(data)

            if data_dict['type'] == 'drawing':
                drawing_data = data_dict['data']
                image_data = data_dict['image_data']

                # Process drawing data and image_data in real-time
                # Perform prediction using the loaded model
                prediction = predict_from_drawing(image_data)

                # Send the prediction result back to the client in real-time
                await websocket.send_text(json.dumps({'type': 'prediction', 'data': prediction}))

    except WebSocketDisconnect:
        pass


def predict_from_drawing(image_data: str):
    try:

        # Extract base64-encoded image data
        _, base64_data = image_data.split(",", 1)
        # Decode the base64 data and convert it to an image
        image = Image.open(io.BytesIO(base64.b64decode(base64_data)))
        # Preprocess the image for prediction
        img_array = preprocess_image(image)
        # Perform prediction using the loaded model
        prediction = model.predict(img_array, verbose=0)
        predicted_class = np.argmax(prediction)
        return {"class": int(predicted_class), "confidence": float(prediction[0][predicted_class])}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")


# Default route to serve the HTML page
@app.get("/")
async def read_root():
    html_file_path = Path("static/index.html")  # Adjust the path accordingly
    return FileResponse(html_file_path)
