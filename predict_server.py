


# from fastapi import FastAPI, File, UploadFile
# from fastapi.middleware.cors import CORSMiddleware
# import uvicorn
# import numpy as np
# from PIL import Image
# import tensorflow as tf
# import io

# app = FastAPI()

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# model = tf.keras.models.load_model("model_new/best_waste_model.h5")

# # optional: import preprocess_input if your model was trained using mobilenetv2 preprocess
# # from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# labels = [
#     "aerosol_cans",
#     "aluminum_food_cans",
#     "aluminum_soda_cans",
#     "cardboard_boxes",
#     "cardboard_packaging",
#     "clothing",
#     "coffee_grounds",
#     "disposable_plastic_cutlery",
#     "eggshells",
#     "food_waste",
#     "glass_beverage_bottles",
#     "glass_cosmetic_containers",
#     "glass_food_jars",
#     "magazines",
#     "newspaper",
#     "office_paper",
#     "paper_cups",
#     "plastic_cup_lids",
#     "plastic_detergent_bottles",
#     "plastic_food_containers",
#     "plastic_shopping_bags",
#     "plastic_soda_bottles",
#     "plastic_straws",
#     "plastic_trash_bags",
#     "plastic_water_bottles",
#     "shoes",
#     "steel_food_cans",
#     "styrofoam_cups",
#     "styrofoam_food_containers",
#     "tea_bags",
# ]

# def get_model_input_size(m):
#     """Return (height, width) expected by the model; fallback to (224,224)."""
#     try:
#         # model.input_shape typically: (None, height, width, channels)
#         shape = m.input_shape
#         if shape and len(shape) == 4 and shape[1] is not None and shape[2] is not None:
#             return int(shape[1]), int(shape[2])
#     except Exception:
#         pass
#     return 224, 224

# @app.post("/predict")
# async def predict(image: UploadFile = File(...)):
#     contents = await image.read()
#     img = Image.open(io.BytesIO(contents)).convert("RGB")  # ensure 3 channels

#     target_h, target_w = get_model_input_size(model)
#     img = img.resize((target_w, target_h))  # PIL uses (width, height) ordering

#     img = np.array(img).astype("float32")

#     # --- Preprocessing: choose the one that matches how you trained the model ---
#     # Option A: if you trained with normalized pixels in [0,1]
#     img = img / 255.0

#     # Option B (alternative): if you trained using MobileNetV2's preprocess_input,
#     # uncomment the import above and use:
#     # img = preprocess_input(img)   # keep pixels in 0-255 before calling preprocess_input

#     # --------------------------------------------------------------------------

#     img = np.expand_dims(img, axis=0)  # (1, H, W, 3)

#     preds = model.predict(img)[0]
#     idx = int(np.argmax(preds))

#     return {
#         "predicted_class": labels[idx] if idx < len(labels) else "unknown",
#         "confidence": float(preds[idx]),
#         "all_probabilities": {labels[i]: float(preds[i]) for i in range(min(len(labels), len(preds)))}
#     }

# if __name__ == "__main__":
#     uvicorn.run(app, host="localhost", port=8000)



from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
from PIL import Image
import tensorflow as tf
import io
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load trained model
model = tf.keras.models.load_model("model_new/best_waste_model.h5")

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
    img = np.array(img).astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)

    preds = model.predict(img)[0]

    # Top-K predictions
    top_k = 3
    top_indices = np.argsort(preds)[-top_k:][::-1]

    top_predictions = [
        {
            "class": labels[i],
            "confidence": float(preds[i])
        }
        for i in top_indices
    ]

    return {
        "primary_class": top_predictions[0]["class"],
        "top_predictions": top_predictions
    }

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)

