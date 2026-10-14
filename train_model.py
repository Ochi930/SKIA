
import os
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# File paths
DATA_FILE = "data/hand_shapes.csv"
MODEL_DIR = "models"
MODEL_FILE = os.path.join(MODEL_DIR, "hand_shape_model.pkl")

# Make sure the dataset exists
if not os.path.exists(DATA_FILE):
    raise FileNotFoundError(
        f"Dataset not found: {DATA_FILE}. "
        "Check that your CSV file is in the data folder."
    )

# Load collected hand landmark data
data = pd.read_csv(DATA_FILE)

if data.empty:
    raise ValueError("The dataset is empty. Collect hand samples first.")

# First column contains the hand-shape label.
# Remaining columns contain the landmark coordinates.
labels = data.iloc[:, 0].astype(str)
features = data.iloc[:, 1:].apply(pd.to_numeric, errors="coerce")

# Remove rows with missing or invalid coordinates
valid_rows = features.notna().all(axis=1)
features = features.loc[valid_rows]
labels = labels.loc[valid_rows]

if len(features) == 0:
    raise ValueError("No valid landmark samples were found.")

# Check the number of examples for each hand shape
print("\nSamples per hand shape:")
print(labels.value_counts())

if labels.nunique() < 2:
    raise ValueError(
        "At least two different hand-shape labels are required."
    )

if labels.value_counts().min() < 2:
    raise ValueError(
        "Each hand shape needs at least two valid samples."
    )

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    features,
    labels,
    test_size=0.20,
    random_state=42,
    stratify=labels
)

# Create and train the classifier
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    class_weight="balanced"
)

print("\nTraining the model...")
model.fit(X_train, y_train)

# Evaluate on samples the model did not train on
predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print(f"\nTest accuracy: {accuracy:.2%}")
print("\nClassification report:")
print(
    classification_report(
        y_test,
        predictions,
        zero_division=0
    )
)

# Save the trained model and its expected input columns
os.makedirs(MODEL_DIR, exist_ok=True)

joblib.dump(
    {
        "model": model,
        "feature_columns": list(features.columns)
    },
    MODEL_FILE
)

print(f"\nModel saved successfully to: {MODEL_FILE}")
print("Milestone 3 training is complete!")
