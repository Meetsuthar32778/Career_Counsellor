"""
model_training.py - Model Training Pipeline
==============================================
Loads the generated career dataset, computes embeddings using
sentence-transformers, trains a RandomForestClassifier, and saves
the trained model.

Usage:
    python -m backend.model.model_training
"""

import os
import pickle
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import classification_report, accuracy_score

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(os.path.dirname(BASE_DIR), "dataset", "career_dataset.csv")
MODEL_PATH = os.path.join(BASE_DIR, "rf_model.pkl")
CLASSES_PATH = os.path.join(BASE_DIR, "classes.txt")


def train_model():
    """
    Full training pipeline with K-Fold cross-validation:
    1. Load dataset
    2. Generate embeddings
    3. Train RandomForest with K-Fold CV
    4. Evaluate on each fold
    5. Train final model on full dataset
    6. Save model
    """
    print("=" * 60)
    print("  AI Career Counsellor - Model Training Pipeline (K-Fold)")
    print("=" * 60)

    # Step 1: Load dataset
    print("\n[Step 1] Loading dataset...")
    df = pd.read_csv(DATASET_PATH)
    print(f"  Loaded {len(df)} samples")
    print(f"  Labels: {df['Label'].unique().tolist()}")
    print(f"  Distribution:\n{df['Label'].value_counts().to_string()}")

    # Step 2: Generate embeddings using sentence-transformers
    print("\n[Step 2] Loading sentence-transformer model...")
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    print("  Model loaded. Generating embeddings...")

    answers = df["Answer"].tolist()
    embeddings = model.encode(answers, show_progress_bar=True, batch_size=256)
    print(f"  Embeddings shape: {embeddings.shape}")

    # Step 3: Prepare data for training
    X = np.array(embeddings)
    y = df["Label"].values

    # Save class labels
    classes = sorted(list(set(y)))
    with open(CLASSES_PATH, "w") as f:
        for cls in classes:
            f.write(cls + "\n")
    print(f"\n[Step 3] Classes saved: {classes}")

    # Step 4: K-Fold Cross-Validation
    print("\n[Step 4] Training RandomForestClassifier with K-Fold CV...")
    kfold = KFold(n_splits=10, shuffle=True, random_state=42)
    fold_accuracies = []
    fold_reports = []

    for fold, (train_idx, test_idx) in enumerate(kfold.split(X), 1):
        print(f"\n  [Fold {fold}/10]")
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        print(f"    Train: {len(X_train)} | Test: {len(X_test)}")

        clf = RandomForestClassifier(
            n_estimators=513,      # Number of trees
            max_depth=None,        # No max depth limit
            min_samples_split=2,
            min_samples_leaf=1,
            random_state=42,
            n_jobs=-1,             # Use all CPU cores
            verbose=0,
        )
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        fold_accuracies.append(accuracy)

        print(f"    Fold Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")
        fold_reports.append(classification_report(y_test, y_pred, output_dict=False))

    # Step 5: Print Cross-Validation Results
    print("\n[Step 5] K-Fold Cross-Validation Results:")
    print(f"  Fold Accuracies: {[f'{acc:.4f}' for acc in fold_accuracies]}")
    print(f"  Mean Accuracy: {np.mean(fold_accuracies):.4f} ({np.mean(fold_accuracies) * 100:.2f}%)")
    print(f"  Std Deviation: {np.std(fold_accuracies):.4f}")

    # Step 6: Train final model on entire dataset
    print("\n[Step 6] Training final model on full dataset...")
    final_clf = RandomForestClassifier(
        n_estimators=513,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=42,
        n_jobs=-1,
        verbose=1,
    )
    final_clf.fit(X, y)
    print("  Final model training complete!")

    # Step 7: Save the model
    print(f"\n[Step 7] Saving model to {MODEL_PATH}...")
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(final_clf, f)
    print("  Model saved successfully!")

    print("\n" + "=" * 60)
    print("  Training Pipeline Complete!")
    print("=" * 60)


if __name__ == "__main__":
    train_model()
