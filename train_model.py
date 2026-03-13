import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib
import os

# Synthetic dataset generation
def generate_data(label, samples=300):
    X, y = [], []
    for _ in range(samples):
        if label == "pushup":
            features = np.random.normal([160, 40, 0.1], [5, 5, 0.05])
        elif label == "situp":
            features = np.random.normal([90, 50, 0.05], [5, 5, 0.03])
        else:  # jump
            features = np.random.normal([170, 120, 0.3], [8, 10, 0.1])

        X.append(features)
        y.append(label)
    return np.array(X), np.array(y)

X1, y1 = generate_data("pushup")
X2, y2 = generate_data("situp")
X3, y3 = generate_data("jump")

X = np.vstack([X1, X2, X3])
y = np.concatenate([y1, y2, y3])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = RandomForestClassifier(n_estimators=150)
model.fit(X_train, y_train)

os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/activity_classifier.joblib")

print("✅ Model trained and saved")
