import os
import numpy as np
import tensorflow as tf
from PIL import Image

from cbam.attention_block import AttentionBlock
from mapping.class_mapping import class_label_map

MODEL_PATH = "../model/model.keras"
IMG_SIZE = (128, 128)

class_names = [
    'actinic keratosis', 'basal cell carcinoma', 'dermatofibroma',
    'melanoma', 'nevus', 'pigmented benign keratosis',
    'seborrheic keratosis', 'squamous cell carcinoma', 'vascular lesion'
]

TEST_DIR = "./melanoma"  

def preprocess_image(path):
    img = Image.open(path).convert("RGB")
    img = img.resize(IMG_SIZE)
    img_array = np.array(img) / 255.0
    return np.expand_dims(img_array, axis=0)

model = tf.keras.models.load_model(MODEL_PATH, custom_objects={"AttentionBlock": AttentionBlock})

for fname in os.listdir(TEST_DIR):
    if fname.lower().endswith(('.jpg', '.png', '.jpeg')):
        img_path = os.path.join(TEST_DIR, fname)
        input_img = preprocess_image(img_path)

        preds = model.predict(input_img)
        class_id = np.argmax(preds)
        print(f"Fichier: {fname}, ID de classe: {class_id}")
        print(f"Prédictions: {preds}")
        confidence = float(np.max(preds))

        class_name = class_names[class_id]

        print(f"\n📄 Image : {fname}")
        print(f"→ Prédiction : {class_name} ({confidence:.3f})")
