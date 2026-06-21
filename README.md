# MoodSense

Mood detection from text. MoodSense classifies any sentence into one of four
moods — **happy, sad, angry, or neutral** — using a fine-tuned DistilBERT
transformer, served behind a Flask API with a lightweight web client.

## Overview

The project pairs two interchangeable models behind a single inference layer:

- **DistilBERT** (fine-tuned on Google's GoEmotions) for accuracy.
- A **TF-IDF + Logistic Regression** model as a fast, dependency-light fallback
  that works without a GPU.

The API automatically uses the transformer when it is available and falls back
to the scikit-learn model otherwise, so the app runs out of the box and upgrades
transparently once the transformer is trained.

GoEmotions' 27 fine-grained emotions are mapped down to the four target moods,
with multi-label examples resolved by majority vote.

## Results

Measured on the GoEmotions test split (4-mood mapping):

| Mood     | scikit-learn (F1) | DistilBERT (F1) |
|----------|:-----------------:|:---------------:|
| happy    | 0.77              | **0.82**        |
| neutral  | 0.70              | **0.74**        |
| sad      | 0.56              | **0.66**        |
| angry    | 0.51              | **0.61**        |
| Accuracy | 0.68              | **0.745**       |
| Macro-F1 | 0.63              | **0.708**       |

Macro-F1 is the headline metric, since neutral dominates the class distribution.

## Quickstart

```bash
pip install -r requirements.txt

# Backend (host 127.0.0.1, port 5000, debug off by default)
python src/api.py

# Frontend (separate terminal)
python -m http.server 5500 --directory frontend
```

Then open http://localhost:5500. Serve the frontend over HTTP rather than opening
the file directly — `file://` origins are rejected by the API's CORS policy.

For GPU training and inference, install the CUDA build of PyTorch first:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu124
```

## Usage

Command line:

```bash
python src/predict.py "i can't stop smiling today"
# Detected Mood: happy  (model: transformer)
```

HTTP:

```bash
curl -X POST http://127.0.0.1:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "this is the worst day ever"}'
# {"mood": "angry"}
```

## Training

```bash
python src/train.py              # scikit-learn model (CPU-friendly)
python src/train_transformer.py  # fine-tune DistilBERT (GPU recommended)
```

Both scripts download GoEmotions, write their artifacts to `models/`, and record
SHA-256 hashes used for integrity verification at load time. The fine-tuned
transformer (~255 MB) is not committed to the repository; regenerate it locally
with `train_transformer.py`.

## Configuration

| Variable          | Default                | Description                        |
|-------------------|------------------------|------------------------------------|
| `HOST`            | `127.0.0.1`            | API bind address                   |
| `PORT`            | `5000`                 | API port                           |
| `FLASK_DEBUG`     | `0`                    | Enable debug mode (development)    |
| `ALLOWED_ORIGINS` | `localhost:5500,:3000` | Comma-separated CORS allowlist     |

## Security

- Debug mode is disabled by default and the server binds to localhost unless
  explicitly overridden.
- `/predict` validates its input and returns generic errors without exposing
  stack traces.
- CORS is restricted to an explicit allowlist rather than a wildcard.
- Predictions are rendered client-side via `textContent` against a fixed mood
  whitelist, preventing HTML injection.
- Model files are verified against recorded SHA-256 hashes before deserialization.

## Tech stack

Python · Flask · scikit-learn · PyTorch · Hugging Face Transformers · GoEmotions
