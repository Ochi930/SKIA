
import joblib
from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

MODEL_PATH = Path("models/hand_shape_model.pkl")
VALIDATION_PATH = Path("data/hand_shapes_validation.csv")

# Check that both files exist.
if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Trained model not found: {MODEL_PATH}")

if not VALIDATION_PATH.exists():
    raise FileNotFoundError(
        f"Validation dataset not found: {VALIDATION_PATH}"
    )

# Load the existing model. This script does not retrain it.
saved_data = joblib.load(MODEL_PATH)

model = saved_data["model"]
feature_columns = saved_data["feature_columns"]

# Load validation data.
data = pd.read_csv(VALIDATION_PATH)

if "label" not in data.columns:
    raise ValueError("Validation CSV must contain a 'label' column.")

if data.empty:
    raise ValueError("Validation dataset is empty.")

# Check that the validation file contains the required features.
missing_features = [
    column for column in feature_columns
    if column not in data.columns
]

if missing_features:
    raise ValueError(
        f"Validation dataset is missing features: {missing_features}"
    )

# Keep the same feature order used during training.
X_validation = data[feature_columns]
y_true = data["label"]

if X_validation.isnull().any().any():
    raise ValueError(
        "Validation data contains missing feature values."
    )

if y_true.isnull().any():
    raise ValueError("Some validation samples have missing labels.")

# Check that both expected classes are present.
expected_labels = ["FIST", "OPEN_PALM"]
actual_labels = sorted(y_true.unique().tolist())

print("\n========== DATASET SUMMARY ==========")
print(f"Total validation samples: {len(data)}")
print("\nSamples per class:")
print(y_true.value_counts().sort_index().to_string())

if actual_labels != sorted(expected_labels):
    print(
        "\nWARNING: Expected labels are FIST and OPEN_PALM. "
        f"Found: {actual_labels}"
    )

# Predict without fitting or changing the model.
y_pred = model.predict(X_validation)

accuracy = accuracy_score(y_true, y_pred)

print("\n========== MODEL EVALUATION ==========")
print(f"Validation accuracy: {accuracy * 100:.2f}%")
print(
    f"Correct predictions: {(y_true == y_pred).sum()} "
    f"out of {len(y_true)}"
)
print(f"Incorrect predictions: {(y_true != y_pred).sum()}")

print("\n========== CLASSIFICATION REPORT ==========")
print(
    classification_report(
        y_true,
        y_pred,
        labels=expected_labels,
        zero_division=0,
    )
)

print("\n========== CONFUSION MATRIX ==========")
print("Rows = actual labels; columns = predicted labels")
print("Label order: FIST, OPEN_PALM")

matrix = confusion_matrix(
    y_true,
    y_pred,
    labels=expected_labels,
)

print(pd.DataFrame(
    matrix,
    index=["Actual FIST", "Actual OPEN_PALM"],
    columns=["Predicted FIST", "Predicted OPEN_PALM"],
).to_string())

# Show a few misclassified examples, if any.
mistakes = data.loc[y_true != y_pred, ["label"]].copy()
mistakes["predicted_label"] = y_pred[y_true != y_pred]

if not mistakes.empty:
    print("\n========== MISCLASSIFIED SAMPLES ==========")
    print("First 20 incorrect predictions:")
    print(mistakes.head(20).to_string(index=True))
else:
    print("\nAll validation samples were classified correctly.")

print("\nEvaluation complete. The trained model was not modified.")
