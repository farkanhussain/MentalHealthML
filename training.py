# ============================================================
# training.py — trains the multi-head mental health model
# ============================================================
import numpy as np
from sklearn.model_selection import train_test_split
from preprocessing import preprocess
from mental_health_model import build_model, compile_model, train_model
from sklearn.metrics import confusion_matrix
import pandas as pd
import numpy as np

# ============================================================
# 1. Load + preprocess dataset
# ============================================================

DATASET_PATH = "mental_health2425v8in.csv"

print("Loading and preprocessing dataset...")
X, y_phq9, y_gad7, y_acha, y_general = preprocess(DATASET_PATH)

# ============================================================
# 2. Train/test split
# ============================================================

print("Splitting dataset...")
X_train, X_test, \
y_phq9_train, y_phq9_test, \
y_gad7_train, y_gad7_test, \
y_acha_train, y_acha_test, \
y_general_train, y_general_test = train_test_split(
    X, y_phq9, y_gad7, y_acha, y_general,
    test_size=0.2,
    random_state=42
)

# ============================================================
# 3. Build + compile model
# ============================================================

input_dim = X.shape[1]
num_disorders = 10

print(f"Building model with input_dim={input_dim}, num_disorders={num_disorders}...")
model = build_model(input_dim, num_disorders)

print("Compiling model...")
model = compile_model(model)

# ============================================================
# 4. Train model
# ============================================================

print("Training model...")
history = train_model(
    model,
    X_train,
    y_phq9_train,
    y_gad7_train,
    y_acha_train,
    y_general_train,
    epochs=50
)

# ============================================================
# 5. Evaluate model
# ============================================================

print("Evaluating model...")
evaluation = model.evaluate(
    X_test,
    {
        "phq9_severity": y_phq9_test,
        "gad7_severity": y_gad7_test,
        "acha_risk": y_acha_test,
        "general_diagnosis": y_general_test
    }
)

print("Evaluation results:", evaluation)

# ============================================================
# 5b. Confusion Matrix 
# ============================================================


print("Generating confusion matrix...")

# Run predictions
phq_pred, gad_pred, acha_pred, diag_pred = model.predict(X_test)

# Convert softmax → predicted class index
y_pred = np.argmax(diag_pred, axis=1)

# Build confusion matrix
cm = confusion_matrix(y_general_test, y_pred)
print("Raw confusion matrix:")
print(cm)

# Convert confusion matrix into TP/FP/FN/TN per class
num_classes = cm.shape[0]
metrics = []

for cls in range(num_classes):
    TP = cm[cls, cls]
    FP = cm[:, cls].sum() - TP
    FN = cm[cls, :].sum() - TP
    TN = cm.sum() - (TP + FP + FN)

    metrics.append({
        "Class": cls,
        "True Positive": TP,
        "False Positive": FP,
        "False Negative": FN,
        "True Negative": TN
    })

df_metrics = pd.DataFrame(metrics)
print("\nPer-class confusion table:")
print(df_metrics.to_markdown())

# ============================================================
# 5c. Overall Accuracy from Confusion Matrix
# ============================================================

# Total correct predictions (sum of diagonal)
total_TP = np.trace(cm)

# Total samples (sum of all confusion matrix entries)
total_samples = cm.sum()

# Accuracy percentage
accuracy = (total_TP / total_samples) * 100

print(f"\nOverall classification accuracy: {accuracy:.2f}%")


# ============================================================
# 6. Save trained model
# ============================================================

print("Saving model...")
model.save("deep_session_multimodel.h5")

print("Training complete. Model saved as deep_session_multimodel.h5")
