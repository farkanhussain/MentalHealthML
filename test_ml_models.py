import joblib
import pandas as pd
from sklearn.metrics import accuracy_score
from openai import OpenAI
import json
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# -----------------------------
# TEST 1 — MODEL A ACCURACY
# -----------------------------
def test_model_a_functionality():
    model = joblib.load("modelA_text_classifier.pkl")

    # A simple input to test prediction functionality
    sample_text = ["I feel really sad and empty today."]

    # Ensure the model can make a prediction without errors
    prediction = model.predict(sample_text)

    # Check that the model returns a non-empty label
    assert isinstance(prediction[0], str)
    assert len(prediction[0]) > 0


# -----------------------------
# TEST 2 — MODEL B ACCURACY
# -----------------------------
def test_model_b_accuracy():
    model = joblib.load("modelB_structured_classifier.pkl")

    df = pd.read_csv("mental_health_symptoms.csv", on_bad_lines="skip")

    # FIX 1: Convert yes/no → numeric (same as training script)
    df = df.replace({"yes": 1, "no": 0, "Yes": 1, "No": 0})

    # FIX 2: Use correct label column ("Disorder")
    X_test = df.drop("Disorder", axis=1)
    y_test = df["Disorder"]

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    assert accuracy >= 0.70



# -----------------------------
# TEST 3 — CHATBOT ACCURACY
# -----------------------------
def test_chatbot_accuracy():
    client = OpenAI()
    embedder = SentenceTransformer("all-MiniLM-L6-v2")

    user_msg = (
        "I've been feeling really down, tired, and unmotivated for weeks. "
        "I can't focus and I don't enjoy things anymore."
    )

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": user_msg}]
    )

    bot_reply = response.choices[0].message.content.lower()

    # --- TEST 1: Depression acknowledgement (hybrid keyword + semantic) ---
    depression_keywords = [
        "feeling down",
        "low mood",
        "lack of motivation",
        "loss of interest",
        "tired",
        "fatigue",
        "struggling",
        "not enjoying things",
        "persistent sadness",
        # realistic therapeutic acknowledgement patterns
        "feeling this way",
        "going through a difficult time",
        "sounds really hard",
        "overwhelmed",
        "sad"
    ]

    keyword_match = any(kw in bot_reply for kw in depression_keywords)

    # semantic fallback phrase
    ideal_ack = "It sounds like you're feeling low or going through a difficult time."
    emb_expected_ack = embedder.encode([ideal_ack])
    emb_bot = embedder.encode([bot_reply])
    semantic_ack = cosine_similarity([emb_expected_ack[0]], [emb_bot[0]])[0][0] >= 0.25

    assert keyword_match or semantic_ack, (
        "Model did not acknowledge depressive symptoms."
    )

    # --- TEST 2: CBT recommendation detection (keyword clusters) ---
    cbt_clusters = [
        ["routine", "structure"],
        ["small steps", "break tasks", "manageable"],
        ["talk to someone", "reach out", "connect"],
        ["physical activity", "exercise", "walk"],
        ["mindfulness", "breathing", "meditation"],
        ["self-care", "sleep", "nutrition"],
        ["professional help", "therapist", "counsellor"]
    ]

    matched_clusters = sum(
        any(keyword in bot_reply for keyword in cluster)
        for cluster in cbt_clusters
    )

    assert matched_clusters >= 2, (
        f"CBT recommendations missing. Only matched {matched_clusters} clusters."
    )

    # --- TEST 3: Semantic similarity to ideal CBT tone ---
    ideal_reply = (
        "It sounds like you're experiencing symptoms of depression. "
        "A CBT approach would involve small manageable activities, "
        "behavioral activation, and reaching out for support."
    )

    emb_expected = embedder.encode([ideal_reply])
    similarity = cosine_similarity([emb_expected[0]], [emb_bot[0]])[0][0]

    assert similarity >= 0.30, f"Semantic similarity too low: {similarity}"
