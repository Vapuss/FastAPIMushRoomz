from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from fastapi.responses import FileResponse

import numpy as np
import json
import uvicorn
import os

app = FastAPI()

@app.get("/")
def read_root():
    return {"status": "online!"}


# 🧠 Încarcă modelul și datele JSON
model = load_model("model/model_complete.keras")


with open("model/specii_ciuperci.json", "r", encoding="utf-8") as f:
    ciuperci_info = json.load(f)

# ✅ Pentru CORS - acceptăm orice frontend momentan
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 🔍 Funcție pentru normalizare imagine

from PIL import ImageEnhance

def preprocess_image(file) -> np.ndarray:
    import io
    from PIL import Image
    import numpy as np

    image_bytes = file.read()
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((224, 224))

    # ⚠️ test: forțăm contrast și luminozitate
    img = ImageEnhance.Contrast(img).enhance(1.4)
    img = ImageEnhance.Brightness(img).enhance(1.2)

    img_array = np.array(img)
    img_array = img_array
    img_array = np.expand_dims(img_array, axis=0)
    return img_array




# 🔎 Găsește detalii despre ciupercă în JSON
def find_mushroom_info(class_name: str):
    # Normalizează denumirea: cu _ devine cu spații, litere mici
    normalized = class_name.replace("_", " ").lower()
    
    for entry in ciuperci_info:
        entry_name = entry["nume"].replace("_", " ").lower()
        if normalized == entry_name:
            return {
                "specie": entry["nume"],
                "denumiri_populare": entry.get("denumire_populara", []),
                "categorie": entry.get("categorie", [])
            }
    
    return {"specie": class_name, "mesaj": "Specie necunoscută în baza JSON."}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    img_data = preprocess_image(file.file)
    prediction = model.predict(img_data)  # -> shape: (1, 110)

    # 🏷️ Încarcă labels
    labels_path = "model/labels.txt"
    labels = []
    if os.path.exists(labels_path):
        with open(labels_path, "r", encoding="utf-8") as f:
            labels = [line.strip() for line in f.readlines()]

    # 🏆 Extrage top 5 indexi
    top_5_indices = prediction[0].argsort()[-5:][::-1]
    top_5_results = []

    print("🔍 Top 5 predicții:")
    for idx in top_5_indices:
        specie = labels[idx] if labels else f"Index {idx}"
        scor = float(round(prediction[0][idx] * 100, 2))
        print(f"{specie}: {scor}%")
        
        info = find_mushroom_info(specie)
        top_5_results.append({
            "specie": info["specie"],
            "scor": scor,
            "denumiri_populare": info.get("denumiri_populare", []),
            "categorie": info.get("categorie", []),
        })

    # 🧠 Verificăm scorul cel mai mare
    top_prediction = top_5_results[0]
    if top_prediction["scor"] > 90.0:
        return {
            "predictie": {
                "specie": top_prediction["specie"],
                "scor": top_prediction["scor"],
                "denumiri_populare": top_prediction["denumiri_populare"],
                "categorie": top_prediction["categorie"]
            }
        }

    return {
        "predictie": {
            "top_5": top_5_results
        }
    }
