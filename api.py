import io

import torch
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

from inference import (
    CHECKPOINT_PATH,
    CLASSIFIER_CHECKPOINT_PATH,
    load_classifier,
    load_segmentation_model,
    run_full_analysis,
)

# Safety net: images with almost no detected burn area, or a low-confidence
# classification, are treated as invalid (e.g. a photo of a car) rather than forced
# into a degree bucket.
MIN_BURN_AREA_FRACTION = 0.005  # 0.5%
MIN_CONFIDENCE = 0.5  # 50%
INVALID_IMAGE_MESSAGE = "Invalid Image: No burn tissue detected. Please upload a clear medical image."

app = FastAPI(title="BurnSense AI API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[api] Loading models on device: {device}")
segmentation_model = load_segmentation_model(CHECKPOINT_PATH, device)
classifier_model, class_names = load_classifier(CLASSIFIER_CHECKPOINT_PATH, device)
print("[api] Models loaded. Ready to serve requests.")


@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")

    result = run_full_analysis(image, segmentation_model, classifier_model, class_names, device)

    class_probs = result["class_probs"]
    predicted_degree = max(class_probs, key=class_probs.get)
    confidence = class_probs[predicted_degree]
    area_fraction = result["area_fraction"]

    if area_fraction < MIN_BURN_AREA_FRACTION or confidence < MIN_CONFIDENCE:
        return {"status": "error", "message": INVALID_IMAGE_MESSAGE}

    healing_info = result["healing_info"]

    return {
        "status": "success",
        "degree": predicted_degree,
        "confidence": confidence,
        "area": area_fraction,
        "class_probs": class_probs,
        "bsi_score": healing_info["bsi_score"],
        "infection_risk": healing_info["infection_risk"],
        "action_plan": healing_info["action_plan"],
        "estimated_days": healing_info["estimated_days"],
    }
