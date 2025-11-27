from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
from PIL import Image

app = Flask(__name__)
CORS(app)

# Load the trained model
model = tf.keras.models.load_model("Model/best_model.h5")

# IMPORTANT: This list MUST match the order Keras used during training
# This is the same list we used in predict.py when it was working correctly
CLASS_NAMES = [
    "cat",
    "chimpanzee",
    "cow",
    "crow",
    "deer",
    "dog",
    "dolphin",
    "eagle",
    "goat",
    "horse",
    "leopard",
    "lion",
    "owl",
    "parrot",
    "pig",
    "shark",
    "sheep",
    "tiger",
    "whale",
    "wolf",
]

IMG_SIZE = (224, 224)

@app.route("/")
def home():
    return jsonify({"message": "Animal Classification API is running!"})


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image received"}), 400

    file = request.files["image"]

    # Load & preprocess image (SAME as training & predict.py)
    img = Image.open(file).convert("RGB")
    img = img.resize(IMG_SIZE)
    img = np.array(img) / 255.0       # rescale to [0,1]
    img = np.expand_dims(img, axis=0) # shape (1, 224, 224, 3)

    preds = model.predict(img)[0]
    index = int(np.argmax(preds))
    confidence = float(preds[index]) * 100.0

    predicted_class = CLASS_NAMES[index]

    return jsonify({
        "animal": predicted_class,
        "confidence": round(confidence, 2)
    })


if __name__ == "__main__":
    app.run(debug=True)
