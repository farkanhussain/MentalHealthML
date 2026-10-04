import pandas as pd
import numpy as np

# ============================================================
# 1. Load CSV
# ============================================================

def load_dataset(path):
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()

    print("CSV columns:", df.columns.tolist())

    if "phq9_6 NUM" in df.columns:
        df = df.rename(columns={"phq9_6 NUM": "phq9_6NUM"})

    return df


# ============================================================
# 2. Extract PHQ-9 scores
# ============================================================

def extract_phq9(df):
    phq9_cols = [
        "phq9_1NUM", "phq9_2NUM", "phq9_3NUM", "phq9_4NUM",
        "phq9_5NUM", "phq9_6NUM", "phq9_7NUM", "phq9_8NUM", "phq9_9NUM"
    ]
    return df[phq9_cols].astype(float).values


# ============================================================
# 3. Extract GAD-7 scores
# ============================================================

def extract_gad7(df):
    gad7_cols = [
        "gad7_1NUM", "gad7_2NUM", "gad7_3NUM",
        "gad7_4NUM", "gad7_5NUM", "gad7_6NUM", "gad7_7NUM"
    ]
    return df[gad7_cols].astype(float).values


# ============================================================
# 4. Extract ACHA features
# ============================================================
def extract_acha(df):
    features = []

    # -----------------------------
    # 1. Student status (Yes/No → 1/0)
    # -----------------------------
    if "fulltime" in df.columns:
        fulltime_numeric = df["fulltime"].astype(str).str.lower().map({
            "yes": 1, "no": 0
        }).fillna(0)
        features.append(fulltime_numeric.values.reshape(-1, 1))

    # -----------------------------
    # 1b. International student (Yes/No → 1/0)
    # -----------------------------
    if "international" in df.columns:
        international_numeric = df["international"].astype(str).str.lower().map({
            "yes": 1, "no": 0
        }).fillna(0)
        features.append(international_numeric.values.reshape(-1, 1))

    # -----------------------------
    # 1c. Sex (Male/Female → 0/1)
    # -----------------------------
    if "sex" in df.columns:
        sex_numeric = df["sex"].astype(str).str.lower().map({
            "male": 0, "female": 1
        }).fillna(0)
        features.append(sex_numeric.values.reshape(-1, 1))

    # -----------------------------
    # 2. Race (string → binary flag)
    # -----------------------------
    race_cols = [c for c in df.columns if c.startswith("race_")]
    if race_cols:
        race_numeric = df[race_cols].astype(str).fillna("").apply(
            lambda col: col.map(
                lambda x: 1 if str(x).strip().lower() not in ["", "no", "none", "nan"] else 0
            )
        )
        features.append(race_numeric.values)

    # -----------------------------
    # 3. Medical conditions (Yes/No → 1/0)
    # -----------------------------
    med_cols = [
        c for c in df.columns
        if c.startswith("acha_12months_any_") and not c.endswith("NUM")
    ]
    if med_cols:
        med_numeric = df[med_cols].astype(str).apply(
            lambda col: col.str.strip().str.lower().map({
                "yes": 1, "no": 0
            }).fillna(0)
        )
        features.append(med_numeric.values)

    # -----------------------------
    # 4. Emotional frequency (string ranges → numeric scale)
    # -----------------------------
    emo_cols = [c for c in df.columns if c.startswith("acha_12months_times_")]
    if emo_cols:
        emo_map = {
            "never": 0,
            "1-2 times": 1,
            "3-4 times": 2,
            "5-6 times": 3,
            "7-8 times": 4,
            "9-10 times": 5,
            "11 or more times": 6
        }

        emo_numeric = df[emo_cols].astype(str).apply(
            lambda col: col.map(lambda x: emo_map.get(str(x).strip().lower(), 0))
        )

        features.append(emo_numeric.values)

    # -----------------------------
    # 5. Depression history (Yes/No/N/A → 1/0)
    # -----------------------------
    hist_cols = [
        c for c in ["acha_depression", "acha_services_1",
                    "acha_services_2", "acha_services_3"]
        if c in df.columns
    ]
    if hist_cols:
        hist_numeric = df[hist_cols].astype(str).apply(
            lambda col: col.str.strip().str.lower().map({
                "yes": 1,
                "no": 0,
                "n/a": 0
            }).fillna(0)
        )
        features.append(hist_numeric.values)

    return np.concatenate(features, axis=1)


# ============================================================
# 5. Build input vectors
# ============================================================

def build_input_vectors(phq9, gad7, acha):
    return np.concatenate([phq9, gad7, acha], axis=1)


# ============================================================
# 6. Build labels (including diagnosis)
# ============================================================

def build_labels(df):

    # -----------------------------------------
    # FIX: Convert emotional frequency strings → numeric
    # -----------------------------------------
    emo_cols = [c for c in df.columns if c.startswith("acha_12months_times_")]

    emo_map = {
        "never": 0,
        "1-2 times": 1,
        "3-4 times": 2,
        "5-6 times": 3,
        "7-8 times": 4,
        "9-10 times": 5,
        "11 or more times": 6
    }

    df[emo_cols] = df[emo_cols].astype(str).apply(
        lambda col: col.map(lambda x: emo_map.get(str(x).strip().lower(), 0))
    )

    # -----------------------------------------
    # Existing label logic (unchanged)
    # -----------------------------------------
    y_phq9 = df["phq9_COMP"].astype(float).values
    y_gad7 = df["gad7_COMP"].astype(float).values
    y_acha = df["acha_12months_any_comp"].astype(float).values

    general_labels = []

    for i in range(len(df)):
        phq = y_phq9[i]
        gad = y_gad7[i]
        acha = y_acha[i]

        suicidal = df["phq9_9NUM"].iloc[i] >= 2
        emo_sum = df[emo_cols].iloc[i].sum()
        burnout = emo_sum >= 12

        dep = phq >= 10
        anx = gad >= 10
        risk = acha >= 5
        severe = phq >= 15 or gad >= 15

        if suicidal:
            general_labels.append(6)
        elif severe:
            general_labels.append(5)
        elif dep and anx:
            general_labels.append(3)
        elif dep:
            general_labels.append(1)
        elif anx:
            general_labels.append(2)
        elif risk:
            general_labels.append(4)
        elif emo_sum >= 8:
            general_labels.append(8)
        elif burnout:
            general_labels.append(9)
        else:
            general_labels.append(0)

    y_general = np.array(general_labels, dtype=np.int32)

    return y_phq9, y_gad7, y_acha, y_general

# ============================================================
# 7. Full preprocessing pipeline
# ============================================================

def preprocess(path):
    df = load_dataset(path)

    phq9 = extract_phq9(df)
    gad7 = extract_gad7(df)
    acha = extract_acha(df)

    X = build_input_vectors(phq9, gad7, acha)
    y_phq9, y_gad7, y_acha, y_general = build_labels(df)

    return X, y_phq9, y_gad7, y_acha, y_general
