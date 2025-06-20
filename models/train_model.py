import json
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from pathlib import Path

# === CONFIG ===
TRAIN_FILE = Path("train.jsonl")
TEST_FILE = Path("test.jsonl")
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

# === LOAD DATA ===
def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line.strip()) for line in f]

train_data = load_jsonl(TRAIN_FILE)
test_data = load_jsonl(TEST_FILE)

df_train = pd.DataFrame(train_data)
df_test = pd.DataFrame(test_data)

# === LABEL ENCODING ===
label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(df_train["output_text"])
y_test = label_encoder.transform(df_test["output_text"])

# === MODEL TRAINING ===
clf = LogisticRegression(max_iter=500)
clf.fit(X_train, y_train)

# === EVALUATION ===
y_pred = clf.predict(X_test)
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

# === SAVE ARTIFACTS ===
joblib.dump(clf, MODEL_DIR / "model.pkl")
joblib.dump(vectorizer, MODEL_DIR / "vectorizer.pkl")
joblib.dump(label_encoder, MODEL_DIR / "label_encoder.pkl")

print("Training complete. Model and assets saved in 'models/' directory.")
