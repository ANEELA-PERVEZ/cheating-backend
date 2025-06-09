from flask import Flask, request, jsonify
import numpy as np
import cv2
import base64
from tensorflow.keras.models import load_model
import os
import gdown

# Google Sheets Logging
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from datetime import datetime

# Download model from Google Drive if not present
MODEL_PATH = "best_model.keras"
if not os.path.exists(MODEL_PATH):
    file_id = "1WcCf96gAw_ME3LHtVQjRdsTNNqQhWNg5"
"  # Replace with your actual file ID
    gdown.download(f"https://drive.google.com/uc?id={file_id}", MODEL_PATH, quiet=False)

# Load the trained model
model = load_model(MODEL_PATH)
IMG_SIZE = 224

# Google Sheets Setup
scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
creds = ServiceAccountCredentials.from_json_keyfile_name("credentials.json", scope)
client = gspread.authorize(creds)
sheet = client.open("Cheating Logs").sheet1

# Initialize Flask app
app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        if 'image' not in data:
            return jsonify({"error": "No image provided"}), 400

        # Decode base64 image
        image_data = base64.b64decode(data['image'].split(',')[1])
        np_arr = np.frombuffer(image_data, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        # Resize and normalize image
        img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE)) / 255.0
        img_input = np.expand_dims(img_resized, axis=0)

        # Predict
        prediction = model.predict(img_input)[0][0]
        is_cheating = prediction > 0.5

        # Log to Google Sheet
        sheet.append_row([
            str(datetime.now()),
            bool(is_cheating),
            float(prediction)
        ])

        return jsonify({"cheating": bool(is_cheating), "confidence": float(prediction)})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
