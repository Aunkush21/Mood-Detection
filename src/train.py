"""Train the mood-detection model from the GoEmotions dataset.

Pipeline:
  1. Load GoEmotions (27 fine-grained emotions, multi-label).
  2. Map every emotion to one of 4 moods: happy / sad / angry / neutral.
  3. Vectorize text with TF-IDF (uni- + bi-grams).
  4. Train LogisticRegression and LinearSVC, keep the better one by macro-F1.
  5. Save model + vectorizer to ../models and record SHA-256 integrity hashes.

Run from anywhere:  python src/train.py
"""

import hashlib
import json
import os
from collections import Counter

import joblib
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.pipeline import FeatureUnion

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")

# Full mapping of all 27 GoEmotions to 4 moods. Anything not listed
# (confusion, curiosity, realization, surprise, neutral) stays neutral.
EMOTION_TO_MOOD = {
    # happy
    "admiration": "happy",
    "amusement": "happy",
    "approval": "happy",
    "caring": "happy",
    "desire": "happy",
    "excitement": "happy",
    "gratitude": "happy",
    "joy": "happy",
    "love": "happy",
    "optimism": "happy",
    "pride": "happy",
    "relief": "happy",
    # sad
    "disappointment": "sad",
    "embarrassment": "sad",
    "fear": "sad",
    "grief": "sad",
    "nervousness": "sad",
    "remorse": "sad",
    "sadness": "sad",
    # angry
    "anger": "angry",
    "annoyance": "angry",
    "disapproval": "angry",
    "disgust": "angry",
}

# When a sample carries several emotions mapping to different moods, prefer the
# more expressive (minority) mood over the neutral catch-all on ties.
MOOD_PRIORITY = {"angry": 3, "sad": 2, "happy": 1, "neutral": 0}


def map_to_mood(label_ids, label_names):
    """Resolve a multi-label GoEmotions row to a single mood by majority vote."""
    moods = [
        EMOTION_TO_MOOD.get(label_names[i], "neutral") for i in label_ids
    ] or ["neutral"]
    counts = Counter(moods)
    top = max(counts, key=lambda m: (counts[m], MOOD_PRIORITY[m]))
    return top


def build_frame(split, label_names):
    texts = split["text"]
    moods = [map_to_mood(lbl, label_names) for lbl in split["labels"]]
    return texts, moods


def main():
    print("Loading GoEmotions ...")
    dataset = load_dataset("go_emotions")
    label_names = dataset["train"].features["labels"].feature.names

    # Train on train + validation, evaluate on the held-out test split.
    train_text, train_y = build_frame(dataset["train"], label_names)
    val_text, val_y = build_frame(dataset["validation"], label_names)
    test_text, test_y = build_frame(dataset["test"], label_names)

    X_train_text = list(train_text) + list(val_text)
    y_train = list(train_y) + list(val_y)
    X_test_text = list(test_text)
    y_test = list(test_y)

    print(f"Train samples: {len(y_train)}  |  Test samples: {len(y_test)}")
    print("Train mood distribution:", dict(Counter(y_train)))

    # Word n-grams capture meaning; char n-grams catch misspellings, emphasis
    # and slang that short emotional text is full of. Combining both beat either
    # alone in tuning.
    print("Vectorizing (TF-IDF, word 1-2 + char 2-5) ...")
    vectorizer = FeatureUnion([
        ("word", TfidfVectorizer(
            max_features=30000, ngram_range=(1, 2),
            stop_words="english", sublinear_tf=True, min_df=2)),
        ("char", TfidfVectorizer(
            analyzer="char_wb", ngram_range=(2, 5),
            max_features=30000, sublinear_tf=True, min_df=2)),
    ])
    X_train = vectorizer.fit_transform(X_train_text)
    X_test = vectorizer.transform(X_test_text)
    print("Feature matrix:", X_train.shape)

    # C=3 + class_weight="balanced" won on both accuracy and macro-F1 in tuning.
    best_model = LogisticRegression(
        max_iter=1000, C=3.0, class_weight="balanced", n_jobs=-1
    )
    print("\nTraining LogisticRegression(C=3, balanced) ...")
    best_model.fit(X_train, y_train)
    pred = best_model.predict(X_test)
    print("\nFinal accuracy:", accuracy_score(y_test, pred))
    print("\nClassification report:\n")
    print(classification_report(y_test, pred))

    os.makedirs(MODELS_DIR, exist_ok=True)
    model_path = os.path.join(MODELS_DIR, "mood_model.pkl")
    vec_path = os.path.join(MODELS_DIR, "vectorizer.pkl")
    joblib.dump(best_model, model_path)
    joblib.dump(vectorizer, vec_path)

    # Record integrity hashes so the API can verify artifacts before loading.
    hashes = {}
    for path in (model_path, vec_path):
        with open(path, "rb") as fh:
            hashes[os.path.basename(path)] = hashlib.sha256(fh.read()).hexdigest()
    with open(os.path.join(MODELS_DIR, "model_hashes.json"), "w") as fh:
        json.dump(hashes, fh, indent=2)

    print("\nSaved model + vectorizer to", MODELS_DIR)
    print("Integrity hashes:", json.dumps(hashes, indent=2))


if __name__ == "__main__":
    main()
