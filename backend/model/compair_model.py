"""
compair_model.py - Model Comparison Pipeline
============================================
Uses the same career dataset and sentence-transformer embeddings to compare:
1. RandomForestClassifier
2. Linear SVM (LinearSVC)
3. Logistic Regression

Usage:
    python -m backend.model.compair_model
"""

import os
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    precision_recall_fscore_support,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(os.path.dirname(BASE_DIR), "dataset", "career_dataset.csv")
RESULTS_PATH = os.path.join(BASE_DIR, "comparison_results.csv")


def run_model_comparison():
    """Run k-fold comparison across multiple classifiers on the same embeddings."""
    print("=" * 72)
    print("  AI Career Counsellor - Classifier Comparison (K-Fold)")
    print("=" * 72)

    # Step 1: Load dataset
    print("\n[Step 1] Loading dataset...")
    df = pd.read_csv(DATASET_PATH)
    print(f"  Loaded {len(df)} samples")
    print(f"  Labels: {df['Label'].unique().tolist()}")
    print(f"  Distribution:\n{df['Label'].value_counts().to_string()}")

    # Step 2: Generate embeddings
    print("\n[Step 2] Generating sentence embeddings...")
    sentence_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    answers = df["Answer"].tolist()
    X = np.array(sentence_model.encode(answers, show_progress_bar=True, batch_size=256))
    y = df["Label"].values
    print(f"  Embeddings shape: {X.shape}")

    # Step 3: Configure models and CV strategy
    print("\n[Step 3] Configuring models...")
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    models = {
        "Random Forest": RandomForestClassifier(
            n_estimators=513,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            random_state=42,
            n_jobs=-1,
            verbose=0,
        ),
        "Linear SVM": make_pipeline(
            StandardScaler(),
            LinearSVC(random_state=42, max_iter=10000),
        ),
        "Logistic Regression": make_pipeline(
            StandardScaler(),
            LogisticRegression(random_state=42, max_iter=2000),
        ),
    }

    # Step 4: Compare outcomes
    print("\n[Step 4] Running 10-fold comparison...")
    comparison_rows = []

    for name, model in models.items():
        print(f"\n--- {name} ---")

        fold_scores = cross_val_score(
            model,
            X,
            y,
            cv=cv,
            scoring="accuracy",
            n_jobs=1,
        )
        y_pred = cross_val_predict(
            model,
            X,
            y,
            cv=cv,
            n_jobs=1,
        )

        acc = accuracy_score(y, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y,
            y_pred,
            average="weighted",
            zero_division=0,
        )

        print(f"Fold Accuracy: {[f'{score:.4f}' for score in fold_scores]}")
        print(f"Mean Accuracy: {fold_scores.mean():.4f} (+/- {fold_scores.std():.4f})")
        print(f"OOF Accuracy : {acc:.4f}")
        print(f"Weighted P/R/F1: {precision:.4f} / {recall:.4f} / {f1:.4f}")
        print("\nClassification Report:")
        print(classification_report(y, y_pred, zero_division=0))

        comparison_rows.append(
            {
                "Model": name,
                "Mean Fold Accuracy": round(float(fold_scores.mean()), 4),
                "Std Fold Accuracy": round(float(fold_scores.std()), 4),
                "OOF Accuracy": round(float(acc), 4),
                "Weighted Precision": round(float(precision), 4),
                "Weighted Recall": round(float(recall), 4),
                "Weighted F1": round(float(f1), 4),
            }
        )

    # Step 5: Final comparison summary
    results_df = pd.DataFrame(comparison_rows).sort_values(
        by=["Mean Fold Accuracy", "Weighted F1"],
        ascending=False,
    )
    results_df.to_csv(RESULTS_PATH, index=False)

    print("\n" + "=" * 72)
    print("Final Comparison (Ranked)")
    print("=" * 72)
    print(results_df.to_string(index=False))
    print(f"\nSaved comparison table to: {RESULTS_PATH}")


if __name__ == "__main__":
    run_model_comparison()
