from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
from werkzeug.utils import secure_filename
import os

app = Flask(__name__)
model = tf.keras.models.load_model("retina_model.h5")

@app.route("/")
def home():
    return "🩺 Diabetes Detection API is Running"

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    image = request.files["image"]
    filename = secure_filename(image.filename)
    image.save(filename)

    # Preprocess the image (resize, normalize, etc.)
    img = tf.keras.utils.load_img(filename, target_size=(224, 224))
    img_array = tf.keras.utils.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    prediction = model.predict(img_array)
    result = "Diabetic" if prediction[0][0] > 0.5 else "Non-Diabetic"

    os.remove(filename)  # Cleanup uploaded file

    return jsonify({"result": result})

if __name__ == "__main__":
    app.run(debug=True)

