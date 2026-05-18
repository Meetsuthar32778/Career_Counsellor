import os
import pandas as pd
import numpy as np
import joblib
from sentence_transformers import SentenceTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE  # pip install imbalanced-learn
from preprocessing import clean_text

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(os.path.dirname(BASE_DIR), "dataset", "career_dataset.csv")
MODEL_PATH = os.path.join(BASE_DIR, "rf_model_best.joblib")
EMBEDDER_PATH = os.path.join(BASE_DIR, "embedder.joblib")
CLASSES_PATH = os.path.join(BASE_DIR, "classes.txt")

# Load data
df = pd.read_csv(DATASET_PATH)
df = df.dropna().drop_duplicates()

# Handle column differences robustly:
if "Answer" in df.columns and "Label" in df.columns:
    df["combined_text"] = df["Answer"].astype(str)
    y = df["Label"]
else:
    # Assuming columns: answer1, answer2, ... answer10, career_field
    # Skip first if demographic
    answer_cols = [col for col in df.columns if col.startswith("answer") and col != "answer1"]
    df["combined_text"] = df[answer_cols].astype(str).agg(" ".join, axis=1)
    y = df["career_field"]

df["combined_text"] = df["combined_text"].apply(clean_text)

# Generate embeddings
encoder = SentenceTransformer("all-MiniLM-L6-v2")
X = encoder.encode(df["combined_text"].tolist(), show_progress_bar=True)

# Split before SMOTE to avoid data leakage
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Apply SMOTE to training data only
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

# Hyperparameter search
param_dist = {
    'n_estimators': [100, 200, 300, 500],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'class_weight': ['balanced', 'balanced_subsample']
}
rf = RandomForestClassifier(random_state=42)
search = RandomizedSearchCV(rf, param_dist, n_iter=30, cv=5, scoring='f1_macro', n_jobs=-1, random_state=42)
search.fit(X_train_res, y_train_res)

best_rf = search.best_estimator_
print("Best parameters:", search.best_params_)

# Evaluate
y_pred = best_rf.predict(X_test)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Save best model and encoder
joblib.dump(best_rf, MODEL_PATH)
joblib.dump(encoder, EMBEDDER_PATH)

# Save classes as well so predictor can use them
classes = sorted(list(set(y.values)))
with open(CLASSES_PATH, "w") as f:
    for cls in classes:
        f.write(cls + "\n")

class_mapping = {i: label for i, label in enumerate(best_rf.classes_)}
joblib.dump(class_mapping, os.path.join(BASE_DIR, "class_mapping.joblib"))
