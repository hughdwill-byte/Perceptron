import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import joblib
from flask import Flask, request, jsonify

# -----------------------------
# Perceptron Definition
# -----------------------------
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class Perceptron:
    def __init__(self, learning_rate=1e-5, n_iters=10000):
        self.lr = learning_rate
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        y_ = np.where(y > 0, 1, 0)  # map {-1,1} → {0,1}

        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                linear_output = np.dot(x_i, self.weights) + self.bias
                y_predicted = sigmoid(linear_output)
                error = y_[idx] - y_predicted
                self.weights += self.lr * error * x_i
                self.bias += self.lr * error

    def predict(self, X):
        linear_output = np.dot(X, self.weights) + self.bias
        probs = sigmoid(linear_output)
        return np.where(probs >= 0.5, 1, 0)

    def predict_single(self, features):
        x_input = np.array(features, dtype=float)
        linear_output = float(np.dot(x_input, self.weights) + self.bias)
        prob = sigmoid(linear_output)
        pred = 1 if prob >= 0.5 else 0
        return pred, linear_output, prob


# -----------------------------
# TRAINING (Run Once)
# -----------------------------
def train_and_save():
    # Path to dataset (update if dataset is elsewhere in repo)
    dataset_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "synthetic_perceptron_dataset.xlsx"
    )

    df = pd.read_excel(dataset_path)

    X = df.drop("BOUGHT", axis=1).values
    feature_names = df.drop("BOUGHT", axis=1).columns.tolist()
    y = df["BOUGHT"].values

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    x_train, x_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=123
    )

    p = Perceptron(learning_rate=0.0001, n_iters=10000)
    p.fit(x_train, y_train)

    # Save both model and scaler
    joblib.dump(p, "perceptron_model.pkl")
    joblib.dump(scaler, "scaler.pkl")
    joblib.dump(feature_names, "features.pkl")

    print("✅ Model, scaler, and features saved!")


# -----------------------------
# FLASK API
# -----------------------------
app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

try:
    p = joblib.load(os.path.join(BASE_DIR, "perceptron_model.pkl"))
    scaler = joblib.load(os.path.join(BASE_DIR, "scaler.pkl"))
    feature_names = joblib.load(os.path.join(BASE_DIR, "features.pkl"))
    print("✅ Model and scaler loaded!")
except FileNotFoundError:
    p = None
    scaler = None
    feature_names = None
    print("⚠️ No saved model found. Run train_and_save() to generate .pkl files.")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        # Expect JSON with keys matching feature_names
        inputs = [data[name] for name in feature_names]
        sample_scaled = scaler.transform([inputs])[0]
        pred, score, conf = p.predict_single(sample_scaled)
        return jsonify({
            "inputs": dict(zip(feature_names, inputs)),
            "prediction": "YES" if pred == 1 else "NO",
            "raw_score": score,
            "confidence": conf
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route("/features", methods=["GET"])
def get_features():
    return jsonify({"feature_names": feature_names})

# -----------------------------
# MAIN
# -----------------------------
if __name__ == "__main__":
    # Train if models are missing
    if not os.path.exists(os.path.join(BASE_DIR, "perceptron_model.pkl")):
        train_and_save()

    # Render expects PORT from env, fallback to 5000 locally
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
