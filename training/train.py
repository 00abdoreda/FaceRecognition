import os
import pickle
import numpy as np
import cv2
from deepface import DeepFace

DATASET_PATH = "training/dataset"
MODEL_PATH = "model/model.pkl"

known_encodings = []
known_names = []

# 🔥 Create 20 variations from 1 image
def augment_image(img):
    h, w = img.shape[:2]
    variations = []

    for i in range(20):
        aug = img.copy()

        # brightness/contrast variation
        alpha = np.random.uniform(0.7, 1.3)
        beta = np.random.randint(-30, 30)
        aug = cv2.convertScaleAbs(aug, alpha=alpha, beta=beta)

        # random flip
        if np.random.rand() > 0.5:
            aug = cv2.flip(aug, 1)

        # slight blur sometimes
        if np.random.rand() > 0.7:
            aug = cv2.GaussianBlur(aug, (3, 3), 0)

        variations.append(aug)

    return variations


print("🔄 Training started...")

for person_name in os.listdir(DATASET_PATH):
    person_path = os.path.join(DATASET_PATH, person_name)

    if not os.path.isdir(person_path):
        continue

    for image_name in os.listdir(person_path):
        image_path = os.path.join(person_path, image_name)

        img = cv2.imread(image_path)
        if img is None:
            continue

        # 🔥 create 20 augmented versions
        augmented_images = augment_image(img)

        for aug in augmented_images:
            try:
                reps = DeepFace.represent(
                    img_path=aug,
                    model_name="ArcFace",   # better than Facenet
                    enforce_detection=False
                )

                if len(reps) == 0:
                    continue

                embedding = np.array(reps[0]["embedding"], dtype=np.float32)

                # normalize
                embedding = embedding / np.linalg.norm(embedding)

                known_encodings.append(embedding)
                known_names.append(person_name)

            except Exception as e:
                print(f"❌ Error: {e}")

print("✅ Training completed")

data = {
    "encodings": np.array(known_encodings),
    "names": np.array(known_names)
}

os.makedirs("model", exist_ok=True)

with open(MODEL_PATH, "wb") as f:
    pickle.dump(data, f)

print("💾 Model saved!")