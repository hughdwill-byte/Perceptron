import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def unit_step_func(x):
    """Activation function: step function."""
    return np.where(x > 0, 1, 0)


class Perceptron:
    def __init__(self, learning_rate=0.01, n_iters=1000):
        self.lr = learning_rate
        self.n_iters = n_iters
        self.activation_func = unit_step_func
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape

        # initialise parameters
        self.weights = np.zeros(n_features)
        self.bias = 0

        # convert labels from {-1,1} → {0,1}
        y_ = np.where(y > 0, 1, 0)

        # training loop
        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                linear_output = np.dot(x_i, self.weights) + self.bias
                y_predicted = self.activation_func(linear_output)

                # update rule
                update = self.lr * (y_[idx] - y_predicted)
                self.weights += update * x_i
                self.bias += update

    def predict(self, X):
        linear_output = np.dot(X, self.weights) + self.bias
        y_predicted = self.activation_func(linear_output)
        return y_predicted

    def predict_single(self, *features):
        """
        Accepts either:
          - p.predict_single([f1, f2, ..., fN])  # single list/array
          - p.predict_single(f1, f2, ..., fN)    # separate args
        Returns: (prediction (0/1), raw_score, confidence 0-1)
        """
        # Handle both list/array and multiple arguments
        if len(features) == 1 and hasattr(features[0], "__iter__") and not isinstance(features[0], (str, bytes)):
            x_input = np.array(features[0], dtype=float)
        else:
            x_input = np.array(features, dtype=float)

        if x_input.shape[0] != self.weights.shape[0]:
            raise ValueError(f"Expected {self.weights.shape[0]} features, got {x_input.shape[0]}")

        linear_output = float(np.dot(x_input, self.weights) + self.bias)
        y_predicted = int(self.activation_func(linear_output))
        confidence = 1.0 / (1.0 + np.exp(-linear_output))  # sigmoid "confidence"
        return y_predicted, linear_output, confidence


def accuracy(y_true, y_pred):
    return np.sum(y_true == y_pred) / len(y_true)


if __name__ == "__main__":
    # ✅ Load Excel dataset
    df = pd.read_excel(
        r"C:\Users\hugh.williams\OneDrive - Peninsula Grammar\Desktop\Personal & Financial Info\percepton\testset.xlsm"
    )

    # ✅ Features = all except 'BOUGHT'
    X = df.drop("BOUGHT", axis=1).values

    # ✅ Labels = 'BOUGHT'
    y = df["BOUGHT"].values

    # ✅ Split train/test
    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=123
    )

    # ✅ Train Perceptron
    p = Perceptron(learning_rate=0.0001, n_iters=10000)
    p.fit(x_train, y_train)

    # ✅ Predict on test set
    predictions = p.predict(x_test)

    # ✅ Print accuracy
    print("Perceptron classification accuracy:", accuracy(y_test, predictions))

    # ✅ Example custom input (must match feature order in Excel)
    # Features: [GOOGLE AD, 10 spent on site, Male, age 50+, Daylight hours, return customers]
    sample_input = [-1, 1, -1, 1, -1, -1]
    pred, score, conf = p.predict_single(sample_input)

    print("\nCustom Input:", sample_input)
    print("Raw Score:", score)
    print("Confidence (0-1):", conf)
    if pred == 1:
        print("Prediction: YES, they will BUY")
    else:
        print("Prediction: NO, they will NOT BUY")
