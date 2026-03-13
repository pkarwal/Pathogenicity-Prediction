import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_validate, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

# =========================
# Load and preprocess data
# =========================
df = pd.read_csv("new_data.csv")

df["label"] = df["Germline classification"].map({
    "Benign": 0,
    "Pathogenic": 1
})

df = df.drop(columns=["Germline classification"])

X = df.drop("label", axis=1)
y = df["label"]

X_encoded = pd.get_dummies(
    X,
    columns=["cds_from", "cds_to", "aa_from", "aa_to"]
)

# =========================
# Cross-validation setup
# =========================
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scoring = ["accuracy", "precision", "recall", "f1", "roc_auc"]

# Helper function to extract mean CV metrics
def get_cv_means(cv_results):
    return {
        "Accuracy": cv_results["test_accuracy"].mean(),
        "Precision": cv_results["test_precision"].mean(),
        "Recall": cv_results["test_recall"].mean(),
        "F1": cv_results["test_f1"].mean(),
        "ROC_AUC": cv_results["test_roc_auc"].mean()
    }

# =========================
# 1. Logistic Regression
# =========================
lr_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=42
    ))
])

lr_param_grid = {
    "model__C": [0.01, 0.1, 1, 5, 10, 50]
}

lr_grid = GridSearchCV(
    lr_pipeline,
    lr_param_grid,
    cv=5,
    scoring="roc_auc",
    n_jobs=-1
)

lr_grid.fit(X_encoded, y)
best_lr = lr_grid.best_estimator_

lr_results = cross_validate(
    best_lr,
    X_encoded,
    y,
    cv=skf,
    scoring=scoring
)

# =========================
# 2. Random Forest
# =========================
rf_model = RandomForestClassifier(
    n_estimators=300,
    class_weight="balanced",
    random_state=42
)

rf_results = cross_validate(
    rf_model,
    X_encoded,
    y,
    cv=skf,
    scoring=scoring
)

# =========================
# 3. Linear SVM
# =========================
svm_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", SVC(
        kernel="linear",
        class_weight="balanced",
        probability=True,
        random_state=42
    ))
])

svm_param_grid = {
    "model__C": [0.01, 0.1, 1, 5, 10, 50]
}

svm_grid = GridSearchCV(
    svm_pipeline,
    svm_param_grid,
    cv=5,
    scoring="roc_auc",
    n_jobs=-1
)

svm_grid.fit(X_encoded, y)
best_svm = svm_grid.best_estimator_

svm_results = cross_validate(
    best_svm,
    X_encoded,
    y,
    cv=skf,
    scoring=scoring
)

# =========================
# Final comparison table
# =========================
results_df = pd.DataFrame([
    {"Model": "Logistic Regression", **get_cv_means(lr_results)},
    {"Model": "Random Forest", **get_cv_means(rf_results)},
    {"Model": "Linear SVM", **get_cv_means(svm_results)}
])

results_df = results_df.sort_values("ROC_AUC", ascending=False).reset_index(drop=True)

print("\nFinal Cross-Validated Model Comparison:")
print(results_df.round(4))

# =========================
# Best hyperparameters
# =========================
print("\nBest Logistic Regression parameters:")
print(lr_grid.best_params_)

print("\nBest Linear SVM parameters:")
print(svm_grid.best_params_)

# =========================
# Final selected model
# =========================
best_model_name = results_df.loc[0, "Model"]
print("\nSelected best model based on highest ROC_AUC:")
print(best_model_name)


# Visualization of feature importance for the best model
import matplotlib.pyplot as plt

metrics = ["Accuracy", "Precision", "Recall", "F1", "ROC_AUC"]

results_df.set_index("Model")[metrics].plot(
    kind="bar",
    figsize=(10,6)
)

plt.title("Cross-Validated Model Performance Comparison")
plt.ylabel("Score")
plt.xticks(rotation=0)
plt.ylim(0,1)
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig("model_comparison.png", dpi=300, bbox_inches="tight")
plt.show()

# Fit Random Forest on full dataset
rf_model = RandomForestClassifier(
    n_estimators=300,
    class_weight="balanced",
    random_state=42
)

rf_model.fit(X_encoded, y)

# Create importance dataframe
importance_df = pd.DataFrame({
    "Feature": X_encoded.columns,
    "Importance": rf_model.feature_importances_
})

# Sort and keep top 15
top_n = 15
top_features = importance_df.sort_values("Importance", ascending=False).head(top_n)

# Plot
plt.figure(figsize=(10, 6))
plt.barh(top_features["Feature"], top_features["Importance"], edgecolor="black")
plt.xlabel("Feature Importance")
plt.ylabel("Feature")
plt.title("Top 15 Random Forest Feature Importances")
plt.gca().invert_yaxis()  # highest importance at top
plt.tight_layout()

plt.savefig("feature_importance.png", dpi=300, bbox_inches="tight")

plt.show()