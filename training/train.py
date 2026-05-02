import os
import pickle
import numpy as np
import cv2
import tempfile
from collections import Counter
from deepface import DeepFace
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

DATASET_PATH = "training/dataset"
MODEL_PATH = "model/model.pkl"

# -------------------------
# 🔥 Augmentation
# -------------------------
def augment_image(img):
    variations = []
    for _ in range(5):
        aug = img.copy()

        alpha = np.random.uniform(0.9, 1.1)
        beta = np.random.randint(-10, 10)
        aug = cv2.convertScaleAbs(aug, alpha=alpha, beta=beta)

        if np.random.rand() > 0.5:
            aug = cv2.flip(aug, 1)

        variations.append(aug)

    return variations


# -------------------------
# 🔥 Get embedding (OLD DeepFace compatible)
# -------------------------
def get_embedding(img):
    try:
        # ensure correct format
        if img.max() <= 1:
            img = (img * 255).astype("uint8")

        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            path = tmp.name
            cv2.imwrite(path, img)

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
# 🔥 Load dataset
# -------------------------
X, y = [], []

print("🔄 Loading dataset...\n")

for person in os.listdir(DATASET_PATH):
    person_path = os.path.join(DATASET_PATH, person)

    if not os.path.isdir(person_path):
        continue

    print(f"📂 {person}")

    for img_name in os.listdir(person_path):
        img_path = os.path.join(person_path, img_name)

        print(f"   🖼️ {img_name}")

        img = cv2.imread(img_path)

        if img is None:
            print("   ❌ failed to load")
            continue

        # original
        emb = get_embedding(img)
        if emb is not None:
            X.append(emb)
            y.append(person)
            print("   ✅ OK")
        else:
            print("   ❌ no face")

        # augmentation
        for aug in augment_image(img):
            emb = get_embedding(aug)
            if emb is not None:
                X.append(emb)
                y.append(person)

print("\n📊 Summary:", Counter(y))
print("Total samples:", len(X))

if len(X) == 0:
    raise ValueError("❌ No embeddings extracted")

X = np.array(X)
y = np.array(y)

# -------------------------
# 🔥 Train/Test
# -------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# -------------------------
# 🔥 Predict
# -------------------------
def predict(x, threshold=0.5):
    sims = np.dot(X_train, x)
    idx = np.argmax(sims)

    if sims[idx] < threshold:
        return "Unknown"

    return y_train[idx]

# -------------------------
# 🔥 Evaluate
# -------------------------
y_pred = [predict(x) for x in X_test]
accuracy = accuracy_score(y_test, y_pred)

print("🎯 Accuracy:", round(accuracy, 4))

# -------------------------
# 🔥 Save model
# -------------------------
os.makedirs("model", exist_ok=True)

with open(MODEL_PATH, "wb") as f:
    pickle.dump({
        "encodings": X_train,
        "names": y_train
    }, f)

print("💾 Model saved!")