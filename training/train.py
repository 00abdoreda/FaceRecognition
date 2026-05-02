import os
import pickle
import numpy as np
import cv2
from collections import Counter
from deepface import DeepFace
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

DATASET_PATH = "training/dataset"
MODEL_PATH = "model/model.pkl"

# -------------------------
# 🔥 Augmentation (light)
# -------------------------
def augment_image(img):
    variations = []
    for _ in range(5):  # keep small to avoid bias
        aug = img.copy()

        alpha = np.random.uniform(0.9, 1.1)
        beta = np.random.randint(-10, 10)
        aug = cv2.convertScaleAbs(aug, alpha=alpha, beta=beta)

        if np.random.rand() > 0.5:
            aug = cv2.flip(aug, 1)

        variations.append(aug)

    return variations


# -------------------------
# 🔥 Get embedding (SAFE)
# -------------------------
def get_embedding(img):
    try:
        reps = DeepFace.represent(
            img_path=img,  # ✅ correct
            model_name="ArcFace",
            detector_backend="retinaface",  # 🔥 important
            enforce_detection=True
        )

        if not reps:
            return None

        emb = np.array(reps[0]["embedding"], dtype=np.float32)

        # normalize
        emb = emb / np.linalg.norm(emb)

        return emb

    except Exception as e:
        print("❌ Embedding error:", e)
        return None


# -------------------------
# 🔥 Load dataset
# -------------------------
X = []
y = []

print("🔄 Loading dataset...\n")

if not os.path.exists(DATASET_PATH):
    raise ValueError(f"❌ Dataset path not found: {DATASET_PATH}")

for person in os.listdir(DATASET_PATH):
    person_path = os.path.join(DATASET_PATH, person)

    if not os.path.isdir(person_path):
        continue

    print(f"📂 Person: {person}")

    for img_name in os.listdir(person_path):
        img_path = os.path.join(person_path, img_name)

        print(f"   🖼️ {img_name}")

        img = cv2.imread(img_path)

        if img is None:
            print("   ❌ Failed to load image")
            continue

        # ORIGINAL
        emb = get_embedding(img)
        if emb is not None:
            X.append(emb)
            y.append(person)
            print("   ✅ Original OK")
        else:
            print("   ❌ No face detected (original)")

        # AUGMENTED
        for aug in augment_image(img):
            emb = get_embedding(aug)
            if emb is not None:
                X.append(emb)
                y.append(person)

print("\n📊 Dataset summary:")
print("Total samples:", len(X))
print("Classes:", Counter(y))


# -------------------------
# ❌ Stop if empty
# -------------------------
if len(X) == 0:
    raise ValueError("❌ No embeddings extracted. Fix dataset or detection.")


X = np.array(X)
y = np.array(y)


# -------------------------
# 🔥 Train/Test Split
# -------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)


# -------------------------
# 🔥 Prediction (cosine similarity)
# -------------------------
def predict(x, threshold=0.5):
    sims = np.dot(X_train, x)  # cosine similarity

    idx = np.argmax(sims)
    score = sims[idx]

    if score < threshold:
        return "Unknown"

    return y_train[idx]


# -------------------------
# 🔥 Evaluate
# -------------------------
y_pred = []

for x in X_test:
    y_pred.append(predict(x))

accuracy = accuracy_score(y_test, y_pred)

print("\n🎯 Accuracy:", round(accuracy, 4))


# -------------------------
# 🔥 Save model
# -------------------------
os.makedirs("model", exist_ok=True)

with open(MODEL_PATH, "wb") as f:
    pickle.dump({
        "encodings": X_train,
        "names": y_train
    }, f)

print("💾 Model saved →", MODEL_PATH)