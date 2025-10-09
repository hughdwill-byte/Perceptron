import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


def sigmoid(x):
    """sigmoid activation (smooth 0-1)."""
    return 1 / (1 + np.exp(-x))

class Perceptron:
    def __init__(self, learning_rate=1e-5, n_iters=10000):
        self.lr = learning_rate
        self.n_iters = n_iters
        self.activation_func = sigmoid
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
                y_predicted = sigmoid(linear_output)

                error = y_[idx] - y_predicted
                self.weights += self.lr * error * x_i
                self.bias += self.lr * error

    def predict(self, X):
        linear_output = np.dot(X, self.weights) + self.bias
        probs = sigmoid(linear_output)
        return np.where(probs >= 0.5, 1, 0)

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
        prob = sigmoid(linear_output)
        pred = 1 if prob >= 0.5 else 0
        return pred, linear_output, prob


def accuracy(y_true, y_pred):
    return np.sum(y_true == y_pred) / len(y_true)


if __name__ == "__main__":
    # ✅ Load Excel dataset
    df = pd.read_excel(
        r"C:\Users\hugh.williams\OneDrive - Peninsula Grammar\Desktop\Personal & Financial Info\percepton\synthetic_perceptron_dataset.xlsx"
    )

    # ✅ Features = all except 'BOUGHT'
    X = df.drop("BOUGHT", axis=1).values

    #New: ✅ Scale features to [0,1] so numeric and binary mix well
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    # ✅ Labels = 'BOUGHT'
    y = df["BOUGHT"].values

    # ✅ Split train/test
    x_train, x_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=123
    )

    # ✅ Train Perceptron
    p = Perceptron(learning_rate=0.0001, n_iters=10000)
    p.fit(x_train, y_train)

    # ✅ Predict on test set
    predictions = p.predict(x_test)

    # ✅ Print accuracy
    print("Perceptron classification accuracy:", accuracy(y_test, predictions))

    # ✅ Example custom input with both numeric and binary (must match feature order in Excel)
    #\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/
    sample_input = [1, 120, 1, 34, 14, 1]

    # New: scale input using the same scaler
    sample_input_scaled = scaler.transform([sample_input])[0]
    pred, score, conf = p.predict_single(sample_input_scaled)

    print("\nCustom Input:", sample_input)
    print("Raw Score:", score)
    print("Confidence (0-1):", conf)
    if pred == 1:
        print("Prediction: YES, they will BUY")
    else:
        print("Prediction: NO, they will NOT BUY")

# EXTRA EVALUATION CODE (not run by default)
# To run, uncomment the lines below and indent appropriately
# ----- EXTRA EVALUATION METRICS -----
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
from sklearn.model_selection import StratifiedKFold
import numpy as np

# 1) Ensure labels are comparable (map {-1,1} -> {0,1} for y_test)
y_test_bin = np.where(y_test > 0, 1, 0)

# 2) Confusion matrix + precision/recall/F1
cm = confusion_matrix(y_test_bin, predictions)
print("\nConfusion matrix [ [TN, FP], [FN, TP] ]:\n", cm)
# Reference:
# [[TN, FP],
#  [FN, TP]]
# TN = True Negatives: model correctly said "NO BUY"
# FP = False Positives: model incorrectly said "BUY"
# FN = False Negatives: model incorrectly said "NO BUY"
# TP = True Positives: model correctly said "BUY"

print("\nClassification report:\n", classification_report(y_test_bin, predictions, digits=3))
# Reference:
# Precision = TP / (TP + FP)  -> Of predicted "BUY", how many were correct
# Recall    = TP / (TP + FN)  -> Of actual "BUY", how many were caught
# F1       

# 3) ROC-AUC using model probabilities from sigmoid
# (recompute probabilities using the learned weights)
def _sigmoid(z):
    return 1 / (1 + np.exp(-z))

test_linear = np.dot(x_test, p.weights) + p.bias
test_probs  = _sigmoid(test_linear)  # P(y=1)
try:
    auc = roc_auc_score(y_test_bin, test_probs)
    print("ROC-AUC:", round(auc, 4))
except ValueError:
    print("ROC-AUC: not defined (only one class present in y_test)")
# Reference:
# ROC-AUC = 1.0 -> perfect, 0.5 -> random guessing
# Example: AUC=0.85 means 85% chance a random BUY is ranked above a random NO-BUY


# 4) Stratified 5-fold cross-validation (manual, using your Perceptron)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_acc = []
cv_f1 = []

for train_idx, val_idx in skf.split(X, np.where(y > 0, 1, 0)):
    X_tr, X_val = X[train_idx], X[val_idx]
    y_tr, y_val = y[train_idx], y[val_idx]

    # train a fresh model each fold (same hyperparams as your main run)
    p_cv = Perceptron(learning_rate=0.0001, n_iters=10000)
    p_cv.fit(X_tr, y_tr)
    y_val_pred = p_cv.predict(X_val)

    y_val_bin = np.where(y_val > 0, 1, 0)
    acc_fold = np.mean(y_val_bin == y_val_pred)

    # F1 (binary) without importing more: compute manually
    tn, fp, fn, tp = confusion_matrix(y_val_bin, y_val_pred).ravel()
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall    = tp / (tp + fn) if (tp + fn) else 0.0
    f1        = 2*precision*recall / (precision + recall) if (precision + recall) else 0.0

    cv_acc.append(acc_fold)
    cv_f1.append(f1)

print("\n5-fold CV Accuracy: mean =", round(np.mean(cv_acc), 4), "std =", round(np.std(cv_acc), 4))
print("5-fold CV F1:       mean =", round(np.mean(cv_f1), 4), "std =", round(np.std(cv_f1), 4))
# Reference:
# If CV mean ≈ test accuracy -> stable model
# If CV mean >> test accuracy -> probably overfitting
# 5) Simple baselines to sanity-check usefulness

pos_rate = y_test_bin.mean()
maj_class_acc = max(pos_rate, 1 - pos_rate)  # accuracy if you always guess the majority class
print("\nBaseline (majority-class) accuracy on test set:", round(maj_class_acc, 4))
# Reference:
# This is the accuracy you'd get if you ALWAYS predicted the most common class.
# Your perceptron must beat this to be useful.

#---------------------------------------

# ----- FEATURE IMPACT ANALYSIS -----
feature_names = df.drop("BOUGHT", axis=1).columns.tolist()

# 1) Feature importance from weights
print("\n=== Feature Importances (by weight) ===")
for name, weight in zip(feature_names, p.weights):
    print(f"{name:20s}  {weight:+.4f}")

# 2) Normalised absolute importance
abs_weights = np.abs(p.weights)
rel_importance = abs_weights / abs_weights.sum()
print("\n=== Relative Impact (normalised) ===")
for name, imp in zip(feature_names, rel_importance):
    print(f"{name:20s}  {imp*100:.2f}%")

# 3) Simple feature selection test (drop-one analysis)
print("\n=== Drop-One Feature Accuracy Test ===")
base_acc = accuracy(y_test, predictions)
print(f"Base Accuracy (all features): {base_acc:.3f}")

for i, name in enumerate(feature_names):
    # remove one feature
    X_reduced = np.delete(X_scaled, i, axis=1)

    # split and retrain
    x_train_r, x_test_r, y_train_r, y_test_r = train_test_split(
        X_reduced, y, test_size=0.2, random_state=123
    )
    p_r = Perceptron(learning_rate=0.001, n_iters=5000)
    p_r.fit(x_train_r, y_train_r)
    preds_r = p_r.predict(x_test_r)

    acc_r = accuracy(y_test_r, preds_r)
    print(f"Dropped {name:20s} -> Accuracy: {acc_r:.3f} (Δ {acc_r - base_acc:+.3f})")

# ----- FORWARD FEATURE SELECTION -----
print("\n=== Forward Feature Selection ===")

remaining_features = list(range(X_scaled.shape[1]))  # indices of features not yet selected
selected_features = []
best_acc = 0.0

while remaining_features:
    scores = []
    for f in remaining_features:
        # test with current selection + this new feature
        candidate = selected_features + [f]
        X_sub = X_scaled[:, candidate]

        # split
        x_train_s, x_test_s, y_train_s, y_test_s = train_test_split(
            X_sub, y, test_size=0.2, random_state=123
        )

        # train a model
        p_s = Perceptron(learning_rate=0.001, n_iters=5000)
        p_s.fit(x_train_s, y_train_s)
        preds_s = p_s.predict(x_test_s)

        acc_s = accuracy(y_test_s, preds_s)
        scores.append((acc_s, f))

    # find best feature to add
    scores.sort(reverse=True, key=lambda x: x[0])
    best_new_acc, best_feature = scores[0]

    if best_new_acc > best_acc:
        best_acc = best_new_acc
        selected_features.append(best_feature)
        remaining_features.remove(best_feature)
        print(f"Added {feature_names[best_feature]:20s} -> Accuracy: {best_new_acc:.3f}")
    else:
        # no improvement by adding any feature, stop
        break

# Print final selected features
print("\nOptimal feature set:")
for idx in selected_features:
    print(f"- {feature_names[idx]}")
print("Final Accuracy:", round(best_acc, 3))


print("DONE")