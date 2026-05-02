from fastapi import FastAPI, UploadFile, File, HTTPException
import numpy as np
import pickle
import cv2
from deepface import DeepFace

app = FastAPI()

MODEL_PATH = "../model/model.pkl"
THRESHOLD = 0.6   # works well after normalization
SCALE = 0.5       # resize for speed

print("🔄 Loading model...")

with open(MODEL_PATH, "rb") as f:
    data = pickle.load(f)

known_encodings = data["encodings"]
known_names = data["names"]

print(f"✅ Loaded {len(known_names)} faces")


@app.get("/")
def home():
    return {"status": "Face Recognition API running 🚀"}


@app.post("/recognize")
async def recognize(file: UploadFile = File(...)):
    try:
        contents = await file.read()

        if not contents:
            raise HTTPException(status_code=400, detail="Empty file")

        # Decode image
        npimg = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

        if img is None:
            raise HTTPException(status_code=400, detail="Invalid image")

        # 🔥 Resize (speed boost)
        small_img = cv2.resize(img, (0, 0), fx=SCALE, fy=SCALE)

        # Detect faces
        faces = DeepFace.extract_faces(
            img_path=small_img,
            enforce_detection=False,
            detector_backend="opencv"  # fastest
        )

        if len(faces) == 0:
            return {"results": [], "message": "No faces found"}

        results = []

        for face_obj in faces:
            face_img = face_obj["face"]

            reps = DeepFace.represent(
                img_path=face_img,
                model_name="ArcFace",
                enforce_detection=False
            )

            if len(reps) == 0:
                continue

            embedding = np.array(reps[0]["embedding"], dtype=np.float32)

            # ✅ Normalize
            embedding = embedding / np.linalg.norm(embedding)

            # 🔥 FAST distance calculation (vectorized)
            distances = np.linalg.norm(known_encodings - embedding, axis=1)

            best_idx = np.argmin(distances)
            best_distance = distances[best_idx]

            # 🚨 VERY IMPORTANT
            if best_distance > THRESHOLD:
                name = "Unknown"
            else:
                name = known_names[best_idx]

            # Better confidence formula
            confidence = float(1 / (1 + best_distance))

            results.append({
                "name": name,
                "confidence": round(confidence, 3)
            })

        return {"results": results}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))