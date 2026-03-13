import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier

# Load data
df = pd.read_csv("new_data.csv")

df["label"] = df["Germline classification"].map({
    "Benign": 0,
    "Pathogenic": 1
})

df = df.drop(columns=["Germline classification"])

X = df.drop("label", axis=1)
y = df["label"]

# One hot encoding
X_encoded = pd.get_dummies(
    X,
    columns=["cds_from","cds_to","aa_from","aa_to"]
)

# Save column names
joblib.dump(X_encoded.columns.tolist(), "feature_columns.pkl")

# Train model
model = RandomForestClassifier(
    n_estimators=300,
    class_weight="balanced",
    random_state=42
)

model.fit(X_encoded, y)

# Save model
joblib.dump(model, "model.pkl")

print("Model saved successfully")

