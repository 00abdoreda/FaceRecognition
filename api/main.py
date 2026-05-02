import numpy as np
import pickle
import cv2
from fastapi import FastAPI, UploadFile, File, HTTPException
from deepface import DeepFace

app = FastAPI()

MODEL_PATH = "../model/model.pkl"
THRESHOLD = 0.5   # cosine similarity threshold (0.4–0.6 recommended)
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
# 🔥 Helper: get embedding
# -------------------------
def get_embedding(face_img):
    try:
        reps = DeepFace.represent(
            img=face_img,  # ✅ FIXED
            model_name="ArcFace",
            enforce_detection=True
        )

        if not reps:
            return None

        emb = np.array(reps[0]["embedding"], dtype=np.float32)

        # normalize
        emb = emb / np.linalg.norm(emb)

        return emb

    except:
        return None


# -------------------------
# 🔥 Predict (cosine similarity)
# -------------------------
def predict(embedding):
    similarities = np.dot(known_encodings, embedding)  # cosine similarity

    best_idx = np.argmax(similarities)
    best_score = similarities[best_idx]

    if best_score < THRESHOLD:
        return "Unknown", float(best_score)

    return known_names[best_idx], float(best_score)


# -------------------------
# 🔥 API Endpoint
# -------------------------
@app.post("/recognize")
async def recognize(file: UploadFile = File(...)):
    try:
        contents = await file.read()

        if not contents:
            raise HTTPException(status_code=400, detail="Empty file")

        # decode image
        npimg = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

        if img is None:
            raise HTTPException(status_code=400, detail="Invalid image")

        # resize for speed
        small_img = cv2.resize(img, (0, 0), fx=SCALE, fy=SCALE)

        # 🔥 Better detector than opencv
        faces = DeepFace.extract_faces(
            img_path=small_img,
            enforce_detection=False,
            detector_backend="retinaface"  # ✅ MUCH better
        )

        if len(faces) == 0:
            return {"results": [], "message": "No faces found"}

        results = []

        for face_obj in faces:
            face_img = face_obj["face"]

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
        raise HTTPException(status_code=500, detail=str(e))