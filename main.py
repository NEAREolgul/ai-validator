import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow_hub as hub

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageOps

import io

from typing import List

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware


checkpoint_path = './model/plagiarism_convnext/plagiarism_siamese_model_convnext'
image_size = 224

def load_model(checkpoint_path: str):
    print("Loading Model")
    plagiarism_siamese_model = tf.keras.Sequential([
        hub.KerasLayer("https://tfhub.dev/sayakpaul/convnext_base_21k_1k_224_fe/1",
                       trainable=False), # Freeze feature extractor weights
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(128, activation=None), # No activation on final dense layer
        tf.keras.layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1)) # L2 normalize embeddings
    ])

    plagiarism_siamese_model.build([None, image_size, image_size, 3])  # Batch input shape
    plagiarism_siamese_model.compile(
        optimizer=tf.keras.optimizers.Adam(0.001),
        loss=tfa.losses.TripletSemiHardLoss()
    )
    print("Loading Weights")
    plagiarism_siamese_model.load_weights(checkpoint_path)
    plagiarism_siamese_model.predict(np.zeros((2, image_size, image_size, 3), dtype=np.float32))
    print("Weights Loaded")
    return plagiarism_siamese_model

plagiarism_siamese_model = load_model(checkpoint_path)

def check_image_plagiarism(image1, image2, similarity_threshold=0.65):
    embs = plagiarism_siamese_model.predict(tf.convert_to_tensor([image1, image2]))
    distance = float(tf.norm(embs[0] - embs[1]))
    
    print('Distance', distance)
    
    if distance < similarity_threshold:
        return True, distance
    return False, distance

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/check_plagiarism/")
async def check_plagiarism(files: List[UploadFile]):
    if len(files) != 2:
        return {"success": False, "message": "Uploaded file should be 2 images"}
    
    imgs = []
    for file in files[:2]:
        img = np.array(Image.open(io.BytesIO(file.file.read())).resize((image_size,image_size)))[..., :3] / 255.0
        imgs.append(img)
        
    is_plagiarism, distance = check_image_plagiarism(imgs[0], imgs[1])
    
   
    return {"success": True, "plagiarism": is_plagiarism, "distance": distance}
    

#if __name__ == "__main__":
#    load_model(checkpoint_path)