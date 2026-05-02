import numpy as np
import pickle
import cv2
import tempfile
import os
from fastapi import FastAPI, UploadFile, File, HTTPException
from deepface import DeepFace

app = FastAPI()

MODEL_PATH = "../model/model.pkl"
THRESHOLD = 0.5
SCALE = 0.5

print("🔄 Loading model...")

with open(MODEL_PATH, "rb") as f:
    data = pickle.load(f)

known_encodings = data["encodings"]
known_names = data["names"]

print(f"✅ Loaded {len(known_names)} faces")


@app.get("/")
def home():
    return {"status": "Face Recognition API running 🚀"}


# -------------------------
# 🔥 Get embedding
# -------------------------
def get_embedding(face_img):
    try:
        if face_img.max() <= 1:
            face_img = (face_img * 255).astype("uint8")

        face_img = cv2.cvtColor(face_img, cv2.COLOR_RGB2BGR)

        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            path = tmp.name
            cv2.imwrite(path, face_img)

        reps = DeepFace.represent(
            img_path=path,
            model_name="ArcFace",
            detector_backend="retinaface",
            enforce_detection=True
        )

        os.remove(path)

        if not reps:
            return None

        emb = np.array(reps[0]["embedding"], dtype=np.float32)
        emb = emb / np.linalg.norm(emb)

        return emb

    except Exception as e:
        print("❌ Embedding error:", e)
        return None


# -------------------------
# 🔥 Predict
# -------------------------
def predict(embedding):
    sims = np.dot(known_encodings, embedding)
    idx = np.argmax(sims)
    score = sims[idx]

    if score < THRESHOLD:
        return "Unknown", float(score)

    return known_names[idx], float(score)


# -------------------------
# 🔥 Endpoint
# -------------------------
@app.post("/recognize")
async def recognize(file: UploadFile = File(...)):
    try:
        contents = await file.read()

        if not contents:
            raise HTTPException(400, "Empty file")

        npimg = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

        if img is None:
            raise HTTPException(400, "Invalid image")

        img = cv2.resize(img, (0, 0), fx=SCALE, fy=SCALE)

        faces = DeepFace.extract_faces(
            img_path=img,
            detector_backend="retinaface",
            enforce_detection=False
        )

        if len(faces) == 0:
            return {"results": [], "message": "No faces found"}

        results = []

        for face in faces:
            face_img = face["face"]

            emb = get_embedding(face_img)
            if emb is None:
                continue

            name, score = predict(emb)

            results.append({
                "name": name,
                "confidence": round(score, 3)
            })

        return {"results": results}

    except Exception as e:
        raise HTTPException(500, str(e))