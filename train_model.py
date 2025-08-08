import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
import joblib

# Load dataset
data = pd.read_csv("healthcare_dataset.csv")

# Features and target
feature_cols = ["Age", "Gender", "Blood Type", "Medical Condition", "Admission Type"]
target_col = "Billing Amount"

X = data[feature_cols]
y = data[target_col]

# Preprocessing categorical features
categorical_features = ["Gender", "Blood Type", "Medical Condition", "Admission Type"]
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
    ],
    remainder="passthrough"
)

# Pipeline with preprocessor and regressor
model = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", RandomForestRegressor(random_state=42))
])

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model.fit(X_train, y_train)

# Save model
joblib.dump(model, "my_healthcare_model.joblib")

print("Model trained and saved successfully.")
