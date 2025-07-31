import os
import numpy as np
import tensorflow as tf
from PIL import Image

from attention_block import AttentionBlock
from class_mapping import class_label_map

MODEL_PATH = "../model/model7.keras"
IMG_SIZE = (128, 128)

class_names = [
    'actinic keratosis', 'basal cell carcinoma', 'dermatofibroma',
    'melanoma', 'nevus', 'pigmented benign keratosis',
    'seborrheic keratosis', 'squamous cell carcinoma', 'vascular lesion'
]

TEST_DIR = "./pigmented_benign_keratosis"  

def preprocess_image(path):
    img = Image.open(path).convert("RGB")
    img = img.resize((128,128))
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)  # (1,128,128,3)
    return img_array

model = tf.keras.models.load_model(MODEL_PATH, custom_objects={"AttentionBlock": AttentionBlock})

for fname in os.listdir(TEST_DIR):
    if fname.lower().endswith(('.jpg', '.png', '.jpeg')):
        img_path = os.path.join(TEST_DIR, fname)
        input_img = preprocess_image(img_path)

        preds = model.predict(input_img)
        class_id = np.argmax(preds)
        print(f"Fichier: {fname}, ID de classe: {class_id}")
        print(f"Prédictions: {preds}")
        print("Raw output (argmax):", class_id)
        print("Proba softmax:", preds)
        confidence = float(np.max(preds))

        class_name = class_names[class_id]
        print("Liste class_names:", class_names)

        print(f"\n📄 Image : {fname}")
        print(f"→ Prédiction : {class_name} ({confidence:.3f})")
