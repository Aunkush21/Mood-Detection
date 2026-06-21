"""Shared mood-inference layer used by both the API and the CLI.

Prefers the fine-tuned DistilBERT model in models/transformer/ when it exists,
and otherwise falls back to the scikit-learn TF-IDF model. Both the API and CLI
import `predict()` from here so there's a single source of truth.
"""

import hashlib
import json
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")
TRANSFORMER_DIR = os.path.join(MODELS_DIR, "transformer")


def _load_verified(path, name, hashes):
    """Load a pickled artifact only after its SHA-256 matches the recorded hash."""
    import joblib

    expected = hashes.get(name)
    if expected:
        with open(path, "rb") as fh:
            actual = hashlib.sha256(fh.read()).hexdigest()
        if actual != expected:
            raise RuntimeError(f"Integrity check failed for {name}: refusing to load.")
    return joblib.load(path)


class _SklearnPredictor:
    backend = "sklearn"

    def __init__(self):
        hashes = {}
        hash_file = os.path.join(MODELS_DIR, "model_hashes.json")
        if os.path.exists(hash_file):
            with open(hash_file) as fh:
                hashes = json.load(fh)
        self.model = _load_verified(
            os.path.join(MODELS_DIR, "mood_model.pkl"), "mood_model.pkl", hashes
        )
        self.vectorizer = _load_verified(
            os.path.join(MODELS_DIR, "vectorizer.pkl"), "vectorizer.pkl", hashes
        )

    def predict(self, text):
        return str(self.model.predict(self.vectorizer.transform([text]))[0])


class _TransformerPredictor:
    backend = "transformer"

    def __init__(self):
        import torch
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
        )

        self.torch = torch
        # Serve on CPU by default: single short requests don't need the GPU and
        # this keeps inference independent of CUDA availability at runtime.
        self.device = "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(TRANSFORMER_DIR)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            TRANSFORMER_DIR
        ).to(self.device)
        self.model.eval()
        with open(os.path.join(TRANSFORMER_DIR, "labels.json")) as fh:
            self.id2label = {int(k): v for k, v in json.load(fh)["id2label"].items()}

    def predict(self, text):
        enc = self.tokenizer(
            text, truncation=True, max_length=128, return_tensors="pt"
        ).to(self.device)
        with self.torch.no_grad():
            logits = self.model(**enc).logits
        return self.id2label[int(logits.argmax(-1).item())]


_predictor = None


def get_predictor():
    """Lazily build and cache the best available predictor."""
    global _predictor
    if _predictor is None:
        if os.path.isdir(TRANSFORMER_DIR) and os.path.exists(
            os.path.join(TRANSFORMER_DIR, "labels.json")
        ):
            _predictor = _TransformerPredictor()
        else:
            _predictor = _SklearnPredictor()
    return _predictor


def predict(text):
    return get_predictor().predict(text)
