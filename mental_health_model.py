import numpy as np
import tensorflow as tf
from keras import layers, Model

# ============================
# 2. Build input vector
# ============================

def build_input_vector(phq9_scores, gad7_scores, acha_features):
    return np.concatenate([phq9_scores, gad7_scores, acha_features])

# ============================
# 3. Build multi-head model
# ============================

def build_model(input_dim, num_disorders):

    input_layer = layers.Input(shape=(input_dim,))

    # ----- PHQ-9 head -----
    phq9_head = layers.Dense(32, activation='relu')(input_layer)
    phq9_head = layers.Dense(16, activation='relu')(phq9_head)
    phq9_embedding = layers.Dense(8, activation='relu')(phq9_head)
    phq9_output = layers.Dense(1, activation='linear', name="phq9_severity")(phq9_embedding)

    # ----- GAD-7 head -----
    gad7_head = layers.Dense(32, activation='relu')(input_layer)
    gad7_head = layers.Dense(16, activation='relu')(gad7_head)
    gad7_embedding = layers.Dense(8, activation='relu')(gad7_head)
    gad7_output = layers.Dense(1, activation='linear', name="gad7_severity")(gad7_embedding)

    # ----- ACHA head -----
    acha_head = layers.Dense(64, activation='relu')(input_layer)
    acha_head = layers.Dense(32, activation='relu')(acha_head)
    acha_embedding = layers.Dense(16, activation='relu')(acha_head)
    acha_output = layers.Dense(1, activation='linear', name="acha_risk")(acha_embedding)

    # ----- Diagnosis head -----
    combined = layers.Concatenate()([phq9_embedding, gad7_embedding, acha_embedding])
    diag_head = layers.Dense(64, activation='relu')(combined)
    diag_head = layers.Dense(32, activation='relu')(diag_head)

    # ⭐ FIX: output 10 classes (0–9)
    diag_output = layers.Dense(num_disorders, activation='softmax', name="general_diagnosis")(diag_head)

    return Model(
        inputs=input_layer,
        outputs=[phq9_output, gad7_output, acha_output, diag_output]
    )

# ============================
# 4. Compile model
# ============================

def compile_model(model):
    model.compile(
        optimizer='adam',
        loss={
            "phq9_severity": "mse",
            "gad7_severity": "mse",
            "acha_risk": "mse",
            "general_diagnosis": "sparse_categorical_crossentropy"
        },
        loss_weights={
            "phq9_severity": 1.0,
            "gad7_severity": 1.0,
            "acha_risk": 0.5,
            "general_diagnosis": 2.0
        }
    )
    return model

# ============================
# 5. Train model
# ============================

def train_model(model, X, y_phq9, y_gad7, y_acha, y_general, epochs=50):
    return model.fit(
        X,
        {
            "phq9_severity": y_phq9,
            "gad7_severity": y_gad7,
            "acha_risk": y_acha,
            "general_diagnosis": y_general
        },
        epochs=epochs,
        batch_size=32
    )
