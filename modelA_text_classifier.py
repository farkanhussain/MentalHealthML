import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix
import onnx
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import StringTensorType
import joblib


# ============================================================
# 1. LOAD JSON DATASET
# ============================================================

df = pd.read_json("DepressionEmo.json", lines=True)
df = df[['text', 'feelings']].dropna()

print("Dataset columns:")
print(df.columns)


# ============================================================
# 2. REMOVE INVALID RECORDS
# ============================================================

df = df[
    df['feelings'].apply(
        lambda x: isinstance(x, list) and len(x) > 0
    )
].copy()


# ============================================================
# 3. TRAIN/TEST SPLIT
#    Split BEFORE expanding the multiple feelings
#    to prevent data leakage.
# ============================================================

train_df, test_df = train_test_split(
    df,
    test_size=0.2,
    random_state=42
)

print("\nOriginal dataset size:", len(df))
print("Training records:", len(train_df))
print("Testing records:", len(test_df))


# ============================================================
# 4. EXPAND FEELINGS FOR SINGLE-LABEL CLASSIFICATION
# ============================================================

def expand_feelings(dataframe):

    rows = []

    for _, row in dataframe.iterrows():

        text = row['text']
        feelings = row['feelings']

        if isinstance(feelings, list):

            for emotion in feelings:
                rows.append([text, emotion])

    return pd.DataFrame(
        rows,
        columns=['symptoms', 'label']
    )


train_expanded = expand_feelings(train_df)
test_expanded = expand_feelings(test_df)


print("\nExpanded training samples:", len(train_expanded))
print("Expanded testing samples:", len(test_expanded))


# ============================================================
# 5. CREATE TRAINING AND TESTING DATA
# ============================================================

X_train = train_expanded['symptoms']
y_train = train_expanded['label']

X_test = test_expanded['symptoms']
y_test = test_expanded['label']


# ============================================================
# 6. PIPELINE
#    TF-IDF + Logistic Regression
# ============================================================

pipeline = Pipeline([
    ("vectorizer", TfidfVectorizer(stop_words='english')),
    ("classifier", LogisticRegression(max_iter=2000))
])


# ============================================================
# 7. TRAIN MODEL
# ============================================================

pipeline.fit(X_train, y_train)

print("\nModel training completed.")


# ============================================================
# 8. SAVE TRAINED MODEL
# ============================================================

joblib.dump(
    pipeline,
    "modelA_text_classifier.pkl"
)

print("Trained model saved as modelA_text_classifier.pkl")


# ============================================================
# 9. TEST MODEL ON UNSEEN TEST DATA
# ============================================================

y_pred = pipeline.predict(X_test)


# ============================================================
# 10. CLASSIFICATION REPORT
# ============================================================

print("\n==============================")
print("CLASSIFICATION REPORT")
print("==============================")

print(
    classification_report(
        y_test,
        y_pred,
        zero_division=0
    )
)


# ============================================================
# 11. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=pipeline.classes_
)

print("\n==============================")
print("CONFUSION MATRIX")
print("==============================")

print(
    pd.DataFrame(
        cm,
        index=pipeline.classes_,
        columns=pipeline.classes_
    )
)


# ============================================================
# 12. TP / FP / FN / TN FOR EACH CLASS
# ============================================================

metrics = []

for index, class_name in enumerate(pipeline.classes_):

    TP = cm[index, index]

    FP = cm[:, index].sum() - TP

    FN = cm[index, :].sum() - TP

    TN = cm.sum() - (TP + FP + FN)

    total = TP + FP + FN + TN

    accuracy = (TP + TN) / total * 100

    metrics.append({
        "Class": class_name,
        "True Positive": TP,
        "False Positive": FP,
        "False Negative": FN,
        "True Negative": TN,
        "Accuracy (%)": round(accuracy, 2)
    })


metrics_df = pd.DataFrame(metrics)


print("\n==============================")
print("CONFUSION MATRIX METRICS")
print("==============================")

print(
    metrics_df.to_string(index=False)
)


# ============================================================
# 13. OVERALL CLASSIFICATION ACCURACY
# ============================================================

total_correct = np.trace(cm)
total_predictions = cm.sum()

overall_accuracy = (
    total_correct / total_predictions
) * 100

print("\n==============================")
print("OVERALL ACCURACY")
print("==============================")

print(
    f"Overall classification accuracy: "
    f"{overall_accuracy:.2f}%"
)


# ============================================================
# 14. EXPORT TO ONNX
# ============================================================

# Android requires a 2D string tensor:
# [batch_size, 1]

initial_type = [
    ('input', StringTensorType([None, 1]))
]

onnx_model = convert_sklearn(
    pipeline,
    initial_types=initial_type,
    target_opset=15
)


with open("modelA.onnx", "wb") as f:
    f.write(
        onnx_model.SerializeToString()
    )


print("\nModel A exported successfully as modelA.onnx")


# ============================================================
# 15. DISPLAY ONNX OUTPUT INFORMATION
# ============================================================

print("\nONNX output names:")

print(
    [o.name for o in onnx_model.graph.output]
)

for output in onnx_model.graph.output:

    print("Name:", output.name)
    print("Type:", output.type)
    print("----")