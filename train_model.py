"""
train_model.py
---------------
Loads data/gestures.csv, trains a Random Forest classifier to recognize
hand gestures from 21 hand landmarks (x, y, z), evaluates it on a held-out
test set, and saves the trained model to models/gesture_model.pkl.
"""

import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
CSV_PATH = os.path.join("data", "gestures.csv")
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "gesture_model.pkl")

RANDOM_STATE = 42
TEST_SIZE = 0.2
N_ESTIMATORS = 200

LABEL_ALIASES = {
    "OPEN": "OPEN",
    "FIST": "FIST",
    "PEACE": "PEACE",
    "THUMBS_UP": "THUMBS_UP",
    "ZERO": "OPEN",
    "ONE": "FIST",
    "TWO": "PEACE",
    "THREE": "THUMBS_UP",
    "FOUR": "OPEN",
    "FIVE": "FIST",
    "zero": "OPEN",
    "one": "FIST",
    "two": "PEACE",
    "three": "THUMBS_UP",
    "four": "OPEN",
    "five": "FIST",
}


def normalize_labels(df):
    """Normalizes legacy labels to the canonical labels from the README."""
    if "label" not in df.columns:
        raise ValueError("CSV is missing the required 'label' column.")

    original_labels = df["label"].dropna().astype(str)
    unknown_labels = sorted({label for label in original_labels if label not in LABEL_ALIASES})
    if unknown_labels:
        print(f"[WARNING] Found unexpected labels: {unknown_labels}. They will be dropped.")

    normalized = df["label"].astype(str).str.strip()
    df = df.copy()
    df["label"] = normalized.map(LABEL_ALIASES)
    df = df.dropna(subset=["label"])
    return df


def load_data(csv_path):
    """Loads and validates the gesture dataset."""
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(
            f"Could not find '{csv_path}'. Run collect_data.py first to generate it."
        )

    df = pd.read_csv(csv_path)

    if df.empty:
        raise ValueError(f"'{csv_path}' is empty. Collect some gesture samples first.")

    df = normalize_labels(df)

    if df.empty:
        raise ValueError(
            f"'{csv_path}' contains no valid gesture labels. "
            "Collect data using the README labels: OPEN, FIST, PEACE, THUMBS_UP."
        )

    # Drop rows with missing values, if any
    before = len(df)
    df = df.dropna()
    after = len(df)
    if before != after:
        print(f"[WARNING] Dropped {before - after} rows containing missing values.")

    return df


def main():
    try:
        print("[INFO] Loading dataset...")
        df = load_data(CSV_PATH)

        label_counts = df["label"].value_counts()
        print("[INFO] Class distribution:")
        print(label_counts)

        if label_counts.min() < 5:
            print(
                "[WARNING] Some classes have very few samples (<5). "
                "Consider collecting more data for reliable accuracy."
            )

        X = df.drop(columns=["label"])
        y = df["label"]

        print(f"[INFO] Total samples: {len(df)} | Features: {X.shape[1]} | Classes: {sorted(y.unique())}")

        # ------------------------------------------------------------------
        # Train / test split (stratified so each class is proportionally
        # represented in both sets)
        # ------------------------------------------------------------------
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
            stratify=y,
        )

        print(f"[INFO] Train samples: {len(X_train)} | Test samples: {len(X_test)}")

        # ------------------------------------------------------------------
        # Train Random Forest
        # ------------------------------------------------------------------
        print("[INFO] Training Random Forest classifier...")
        model = RandomForestClassifier(
            n_estimators=N_ESTIMATORS,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
        model.fit(X_train, y_train)

        # ------------------------------------------------------------------
        # Evaluate
        # ------------------------------------------------------------------
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)

        print("\n" + "=" * 60)
        print(f"Test Accuracy: {acc * 100:.2f}%")
        print("=" * 60)
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        print("Confusion Matrix:")
        print(confusion_matrix(y_test, y_pred, labels=sorted(y.unique())))

        # ------------------------------------------------------------------
        # Save model
        # ------------------------------------------------------------------
        os.makedirs(MODEL_DIR, exist_ok=True)
        joblib.dump(model, MODEL_PATH)
        print(f"\n[INFO] Model saved to '{MODEL_PATH}'")

    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        sys.exit(1)
    except ValueError as e:
        print(f"[ERROR] {e}")
        sys.exit(1)
    except Exception as e:
        print(f"[ERROR] Unexpected error during training: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
