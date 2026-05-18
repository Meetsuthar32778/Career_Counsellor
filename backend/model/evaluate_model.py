import os
import joblib
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from preprocessing import clean_text

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(os.path.dirname(BASE_DIR), "dataset", "career_dataset.csv")
MODEL_PATH = os.path.join(BASE_DIR, "rf_model_best.joblib")
EMBEDDER_PATH = os.path.join(BASE_DIR, "embedder.joblib")

print("Loading model and encoder...")
rf = joblib.load(MODEL_PATH)
encoder = joblib.load(EMBEDDER_PATH)

print("Loading dataset...")
df = pd.read_csv(DATASET_PATH)
df = df.dropna().drop_duplicates()

if "Answer" in df.columns and "Label" in df.columns:
    df["combined_text"] = df["Answer"].astype(str)
    y = df["Label"]
else:
    answer_cols = [col for col in df.columns if col.startswith("answer") and col != "answer1"]
    df["combined_text"] = df[answer_cols].astype(str).agg(" ".join, axis=1)
    y = df["career_field"]

print("Applying text preprocessing...")
df["combined_text"] = df["combined_text"].apply(clean_text)

print("Generating embeddings...")
X = encoder.encode(df["combined_text"].tolist(), show_progress_bar=True)

print("Making predictions...")
y_pred = rf.predict(X)

print("\nClassification Report (Full Dataset):")
print(classification_report(y, y_pred))
