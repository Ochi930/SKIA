
from pathlib import Path

import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

DATA_FILE = Path("data/hand_shapes.csv")
MODEL_FILE = Path("models/hand_shape_model.pkl")

if not DATA_FILE.exists():
    raise FileNotFoundError(f"Training dataset not found: {DATA_FILE}")

data = pd.read_csv(DATA_FILE)

if data.empty:
    raise ValueError("Training dataset is empty.")

if "label" not in data.columns:
    raise ValueError("Training dataset has no label column.")

labels = data["label"].astype(str)
features = data.drop(columns=["label"]).apply(
    pd.to_numeric, errors="coerce"
)

# Remove rows containing missing or invalid feature values.
valid_rows = features.notna().all(axis=1) & labels.notna()
features = features.loc[valid_rows]
labels = labels.loc[valid_rows]

if features.empty:
    raise ValueError("No valid training samples were found.")

if len(features.columns) != 63:
    raise ValueError(
        f"Expected 63 landmark features, found {len(features.columns)}."
    )

if labels.nunique() < 2:
    raise ValueError("At least two classes are required.")

if labels.value_counts().min() < 2:
    raise ValueError("Each class needs at least two samples.")

print("\nTraining samples per class:")
print(labels.value_counts().sort_index().to_string())
print(f"\nTotal valid training samples: {len(features)}")

# This split is for a training-side diagnostic only.
# The separate validation CSV is not used here.
X_train, X_test, y_train, y_test = train_test_split(
    features,
    labels,
    test_size=0.20,
    random_state=42,
    stratify=labels,
)

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight="balanced",
)

print("\nTraining Random Forest classifier...")
model.fit(X_train, y_train)

predictions = model.predict(X_test)

print(f"\nTraining-side test accuracy: "
      f"{accuracy_score(y_test, predictions):.2%}")

print("\nTraining-side classification report:")
print(classification_report(y_test, predictions, zero_division=0))

# Save the model and the exact feature-column order.
MODEL_FILE.parent.mkdir(parents=True, exist_ok=True)

joblib.dump(
    {
        "model": model,
        "feature_columns": list(features.columns),
    },
    MODEL_FILE,
)

print(f"\nModel saved successfully: {MODEL_FILE}")
print("Training complete.")
