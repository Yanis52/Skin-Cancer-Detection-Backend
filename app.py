

from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import os

from cbam.attention_block import AttentionBlock
from mapping.class_mapping import class_label_map
app = Flask(__name__)

# Chargement du modèle 
MODEL_PATH = "./model/model7.keras"
model = tf.keras.models.load_model(MODEL_PATH, custom_objects={"AttentionBlock": AttentionBlock})


class_names = [
    'kératose actinique', 'carcinome basocellulaire', 'dermatofibrome',
    'mélanome', 'naevus', 'kératose pigmentée bénigne',
    'kératose séborrhéique', 'carcinome épidermoïde', 'lésion vasculaire'
]

# Taille d'image pour le prétraitement
IMG_SIZE = (128, 128)

def preprocess_image(file):
    img = Image.open(file).convert("RGB")
    img = img.resize((128,128))
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)  # (1,128,128,3)
    return img_array


# endpoint de base
@app.route("/", methods=["GET"])
def index():
    return jsonify({"message": "API CBAM opérationnelle"})

# endpoint de prédiction
@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "Aucun fichier n'a été fourni"}), 400

    file = request.files["file"]
    print(f"Fichier reçu: {file.filename}")
    try:
        img_array = preprocess_image(file)
        preds = model.predict(img_array)
        class_id = np.argmax(preds)
        class_name_en = class_names[class_id]
        confidence = float(np.max(preds))
        class_name_fr = class_label_map.get(class_name_en, class_name_en)
        print(f"Prédiction: {class_name_en}, Confiance: {confidence}")
        return jsonify({
            "class": class_name_fr,
            "confidence": round(confidence, 3)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
