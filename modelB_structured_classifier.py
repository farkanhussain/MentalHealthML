import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
import joblib
import onnx
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
import os
import csv

print("Loading ONNX from:", os.path.abspath("modelB.onnx"))

# --- DEBUG: RAW CSV CHECK ---
with open("mental_health_symptoms.csv", "r", encoding="utf-8", errors="ignore") as f:
    first_line = f.readline()
    print("RAW FIRST LINE:", first_line)

with open("mental_health_symptoms.csv", "r", encoding="utf-8", errors="ignore") as f:
    dialect = csv.Sniffer().sniff(f.read(2048))
    print("DETECTED DELIMITER:", repr(dialect.delimiter))

# 1. LOAD DATASET (force only 25 columns)
df = pd.read_csv("mental_health_symptoms.csv", sep=",", usecols=range(25))

# Drop empty columns if any remain
df = df.dropna(axis=1, how="all")



# Clean column names
df.columns = df.columns.str.strip().str.replace(r'\s+', '', regex=True)


# --- 4. FIX MULTIPLE DISORDER COLUMNS IF PRESENT ---
if isinstance(df["Disorder"], pd.DataFrame):
    print("Multiple Disorder columns detected — fixing...")
    df["Disorder"] = df["Disorder"].iloc[:, 0]

print("Disorder type:", type(df["Disorder"]))
print("Disorder shape:", df["Disorder"].shape)

# --- 5. CONVERT yes/no → 1/0 ---
df = df.replace({"yes": 1, "no": 0, "Yes": 1, "No": 0})

print("Columns:", df.columns.tolist())
print("Raw Disorder column preview:")
print(df["Disorder"].head(10))

# --- 6. SEPARATE FEATURES AND LABEL ---
X = df.drop("Disorder", axis=1)

# --- 7. ENCODE LABELS ---
le = LabelEncoder()
y = le.fit_transform(df["Disorder"])
joblib.dump(le, "label_encoder.pkl")

print("Encoded labels:", set(y))
print("y shape:", y.shape)

# --- 8. TRAIN/TEST SPLIT ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# --- 9. TRAIN MODEL ---
model = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    random_state=42
)
model.fit(X_train, y_train)

# --- 10. EVALUATE ---
preds = model.predict(X_test)
print(classification_report(y_test, preds))

# --- 11. SAVE PICKLE ---
joblib.dump(model, "modelB_structured_classifier.pkl")

# --- 12. EXPORT TO ONNX ---
initial_type = [('input', FloatTensorType([None, X.shape[1]]))]



onnx_model = convert_sklearn(
    model,
    initial_types=initial_type,
    target_opset=12
)

onnx.save_model(onnx_model, "modelB.onnx")
print("Model B exported successfully as modelB.onnx")

print("Reloading ONNX from:", os.path.abspath("modelB.onnx"))

# --- 13. VERIFY OUTPUT SHAPES ---
m = onnx.load("modelB.onnx")
print(m.graph.output)
