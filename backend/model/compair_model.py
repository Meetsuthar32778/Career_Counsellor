"""
compair_model.py - Model Comparison Pipeline
============================================
Uses the same career dataset and sentence-transformer embeddings to compare:
1. RandomForestClassifier
2. Linear SVM (LinearSVC)
3. Logistic Regression
4. LSTM Neural Network

Usage:
    python -m backend.model.compair_model
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
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
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import LinearSVC

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(os.path.dirname(BASE_DIR), "dataset", "career_dataset.csv")
RESULTS_PATH = os.path.join(BASE_DIR, "comparison_results.csv")

# Device for PyTorch
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# LSTM Model Definition
# ---------------------------------------------------------------------------
class LSTMClassifier(nn.Module):
    """LSTM-based classifier for career field prediction."""
    
    def __init__(self, input_dim, hidden_dim, num_classes, num_layers=2, dropout=0.3):
        super(LSTMClassifier, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
        )
        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )
    
    def forward(self, x):
        # x shape: (batch_size, 384) -> reshape to (batch_size, 1, 384) for LSTM
        x = x.unsqueeze(1)
        lstm_out, _ = self.lstm(x)
        # Take last timestep
        last_out = lstm_out[:, -1, :]
        logits = self.fc(last_out)
        return logits


def train_lstm_fold(X_train, y_train, X_test, y_test, num_classes, epochs=50):
    """Train LSTM for one fold and return predictions."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Convert to torch tensors
    X_train_t = torch.FloatTensor(X_train_scaled).to(DEVICE)
    y_train_t = torch.LongTensor(y_train).to(DEVICE)
    X_test_t = torch.FloatTensor(X_test_scaled).to(DEVICE)
    
    # Create data loader
    train_dataset = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    # Initialize model
    model = LSTMClassifier(
        input_dim=384,
        hidden_dim=128,
        num_classes=num_classes,
        num_layers=2,
        dropout=0.3,
    ).to(DEVICE)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Train
    model.train()
    for epoch in range(epochs):
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
    
    # Predict
    model.eval()
    with torch.no_grad():
        test_logits = model(X_test_t)
        test_preds = test_logits.argmax(dim=1).cpu().numpy()
    
    return test_preds


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
    
    # Encode labels for LSTM
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    num_classes = len(label_encoder.classes_)
    
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

    # Standard sklearn models
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

    # LSTM Neural Network
    print(f"\n--- LSTM Neural Network ---")
    lstm_fold_scores = []
    lstm_all_preds = np.array([])
    lstm_all_true = np.array([])
    
    for fold, (train_idx, test_idx) in enumerate(cv.split(X, y_encoded), 1):
        print(f"  [Fold {fold}/10]", end=" ")
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y_encoded[train_idx], y_encoded[test_idx]
        
        y_pred = train_lstm_fold(X_train, y_train, X_test, y_test, num_classes, epochs=50)
        fold_acc = accuracy_score(y_test, y_pred)
        lstm_fold_scores.append(fold_acc)
        
        lstm_all_preds = np.append(lstm_all_preds, y_pred)
        lstm_all_true = np.append(lstm_all_true, y_test)
        
        print(f"Accuracy: {fold_acc:.4f}")
    
    lstm_mean_acc = np.mean(lstm_fold_scores)
    lstm_std_acc = np.std(lstm_fold_scores)
    lstm_oof_acc = accuracy_score(lstm_all_true, lstm_all_preds)
    lstm_precision, lstm_recall, lstm_f1, _ = precision_recall_fscore_support(
        lstm_all_true,
        lstm_all_preds,
        average="weighted",
        zero_division=0,
    )
    
    print(f"Fold Accuracy: {[f'{score:.4f}' for score in lstm_fold_scores]}")
    print(f"Mean Accuracy: {lstm_mean_acc:.4f} (+/- {lstm_std_acc:.4f})")
    print(f"OOF Accuracy : {lstm_oof_acc:.4f}")
    print(f"Weighted P/R/F1: {lstm_precision:.4f} / {lstm_recall:.4f} / {lstm_f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(lstm_all_true, lstm_all_preds, zero_division=0))
    
    comparison_rows.append(
        {
            "Model": "LSTM Neural Network",
            "Mean Fold Accuracy": round(float(lstm_mean_acc), 4),
            "Std Fold Accuracy": round(float(lstm_std_acc), 4),
            "OOF Accuracy": round(float(lstm_oof_acc), 4),
            "Weighted Precision": round(float(lstm_precision), 4),
            "Weighted Recall": round(float(lstm_recall), 4),
            "Weighted F1": round(float(lstm_f1), 4),
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
