import json

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

FEATURES = ["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]

df = pd.read_csv("Iris.csv")
X = df[FEATURES]
y = df["Species"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Fit the scaler on training data only, then train the model on scaled data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train_scaled, y_train)

accuracy = accuracy_score(y_test, model.predict(scaler.transform(X_test)))

joblib.dump(model, "iris_model.pkl")
joblib.dump(scaler, "iris_scaler.pkl")
with open("iris_metrics.json", "w") as f:
    json.dump({"model": "K-Nearest Neighbors (k=5)", "test_accuracy": round(accuracy * 100, 2)}, f)

print(f"Saved iris_model.pkl, iris_scaler.pkl, iris_metrics.json. Test accuracy: {accuracy:.2%}")
