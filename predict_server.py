





# from fastapi import FastAPI, File, UploadFile
# from fastapi.middleware.cors import CORSMiddleware
# import uvicorn
# import numpy as np
# from PIL import Image
# import tensorflow as tf
# import io
# import os

# app = FastAPI()

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # Load trained model
# model = tf.keras.models.load_model("model_new/best_waste_model.h5")

# labels = [
#     "aerosol_cans","aluminum_food_cans","aluminum_soda_cans",
#     "cardboard_boxes","cardboard_packaging","clothing",
#     "coffee_grounds","disposable_plastic_cutlery","eggshells",
#     "food_waste","glass_beverage_bottles","glass_cosmetic_containers",
#     "glass_food_jars","magazines","newspaper","office_paper",
#     "paper_cups","plastic_cup_lids","plastic_detergent_bottles",
#     "plastic_food_containers","plastic_shopping_bags",
#     "plastic_soda_bottles","plastic_straws","plastic_trash_bags",
#     "plastic_water_bottles","shoes","steel_food_cans",
#     "styrofoam_cups","styrofoam_food_containers","tea_bags"
# ]

# @app.post("/predict")
# async def predict(image: UploadFile = File(...)):
#     contents = await image.read()

#     img = Image.open(io.BytesIO(contents)).convert("RGB")
#     img = img.resize((224, 224))
#     img = np.array(img).astype("float32") / 255.0
#     img = np.expand_dims(img, axis=0)

#     preds = model.predict(img)[0]

#     # Top-K predictions
#     top_k = 3
#     top_indices = np.argsort(preds)[-top_k:][::-1]

#     top_predictions = [
#         {
#             "class": labels[i],
#             "confidence": float(preds[i])
#         }
#         for i in top_indices
#     ]

#     return {
#         "primary_class": top_predictions[0]["class"],
#         "top_predictions": top_predictions
#     }

# if __name__ == "__main__":
#     port = int(os.environ.get("PORT", 8000))
#     uvicorn.run(app, host="0.0.0.0", port=port)

import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from PIL import Image
import tensorflow as tf
import io

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model_new", "best_waste_model.h5")

# Load model ONCE
model = tf.keras.models.load_model(MODEL_PATH, compile=False)

labels = [
    "aerosol_cans","aluminum_food_cans","aluminum_soda_cans",
    "cardboard_boxes","cardboard_packaging","clothing",
    "coffee_grounds","disposable_plastic_cutlery","eggshells",
    "food_waste","glass_beverage_bottles","glass_cosmetic_containers",
    "glass_food_jars","magazines","newspaper","office_paper",
    "paper_cups","plastic_cup_lids","plastic_detergent_bottles",
    "plastic_food_containers","plastic_shopping_bags",
    "plastic_soda_bottles","plastic_straws","plastic_trash_bags",
    "plastic_water_bottles","shoes","steel_food_cans",
    "styrofoam_cups","styrofoam_food_containers","tea_bags"
]

@app.post("/predict")
async def predict(image: UploadFile = File(...)):
    contents = await image.read()

    img = Image.open(io.BytesIO(contents)).convert("RGB")
    img = img.resize((224, 224))
    img = np.array(img, dtype=np.float32) / 255.0
    img = np.expand_dims(img, axis=0)

    preds = model.predict(img)[0]

    top_k = 3
    top_indices = np.argsort(preds)[-top_k:][::-1]

    return {
        "primary_class": labels[top_indices[0]],
        "top_predictions": [
            {"class": labels[i], "confidence": float(preds[i])}
            for i in top_indices
        ]
    }
